from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from urllib.parse import unquote, urlparse


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_DIR / "storage" / "kb.sqlite3"
MYSQL_ENV_PATH = PROJECT_DIR / "deploy" / "mysql" / ".env"
MYSQL_SCHEMA_PATH = PROJECT_DIR / "deploy" / "mysql" / "init" / "001_product_catalog.sql"
_MYSQL_POOL = None


class HybridRow(dict):
    """Mapping row compatible with sqlite3.Row's name and numeric access."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


def _compatible_value(key: str, value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if key.endswith("_json") and value is not None and not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return value


class MySQLCursorAdapter:
    def __init__(self, cursor):
        self.cursor = cursor

    @property
    def rowcount(self) -> int:
        return int(self.cursor.rowcount)

    def _row(self, row):
        if row is None:
            return None
        if isinstance(row, dict):
            return HybridRow({key: _compatible_value(key, value) for key, value in row.items()})
        return row

    def fetchone(self):
        return self._row(self.cursor.fetchone())

    def fetchall(self):
        return [self._row(row) for row in self.cursor.fetchall()]


def _mysql_sql(query: str) -> str:
    import re

    converted = query.replace("?", "%s")
    converted = re.sub(
        r"ON\s+CONFLICT\s*\([^)]*\)\s+DO\s+UPDATE\s+SET",
        "ON DUPLICATE KEY UPDATE",
        converted,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    converted = re.sub(
        r"excluded\.([a-zA-Z_][a-zA-Z0-9_]*)",
        r"VALUES(\1)",
        converted,
        flags=re.IGNORECASE,
    )
    converted = re.sub(r"\s+AS\s+INTEGER\)", " AS SIGNED)", converted, flags=re.IGNORECASE)
    return converted


def _mysql_param(value: Any) -> Any:
    if isinstance(value, str) and "T" in value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                from datetime import timezone

                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except ValueError:
            return value
    return value


class MySQLConnectionAdapter:
    backend = "mysql"

    def __init__(self, connection):
        self.connection = connection

    def execute(self, query: str, params=()):
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(_mysql_sql(query), tuple(_mysql_param(value) for value in (params or ())))
        return MySQLCursorAdapter(cursor)

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def close(self) -> None:
        self.connection.close()


def _read_env(path: Path = MYSQL_ENV_PATH) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def product_database_backend() -> str:
    return (os.getenv("PRODUCT_DB_BACKEND") or _read_env().get("PRODUCT_DB_BACKEND") or "sqlite").lower()


def product_database_url() -> str:
    return os.getenv("PRODUCT_DATABASE_URL") or _read_env().get("PRODUCT_DATABASE_URL", "")


def _mysql_connect():
    global _MYSQL_POOL
    try:
        import mysql.connector
        from mysql.connector.pooling import MySQLConnectionPool
    except ImportError as exc:
        raise RuntimeError("缺少mysql-connector-python，无法连接产品数据库") from exc
    parsed = urlparse(product_database_url())
    if parsed.scheme != "mysql" or not parsed.hostname or not parsed.path.strip("/"):
        raise RuntimeError("PRODUCT_DATABASE_URL配置不正确")
    if _MYSQL_POOL is None:
        _MYSQL_POOL = MySQLConnectionPool(
            pool_name="mambadfuse_product_catalog",
            pool_size=10,
            pool_reset_session=True,
            host=parsed.hostname,
            port=int(parsed.port or 3306),
            user=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=parsed.path.strip("/"),
            charset="utf8mb4",
            autocommit=False,
            connection_timeout=10,
        )
    connection = _MYSQL_POOL.get_connection()
    return MySQLConnectionAdapter(connection)


@contextmanager
def connect(db_path: Optional[Path] = None) -> Iterator[Any]:
    if db_path is None and product_database_backend() == "mysql":
        conn = _mysql_connect()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return

    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


MIGRATION_001 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS manufacturers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    website TEXT,
    country_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_url TEXT,
    source_sha256 TEXT NOT NULL,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    retrieved_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(document_title, source_sha256, page_number)
);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manufacturer_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    name TEXT NOT NULL,
    modality TEXT NOT NULL CHECK (
        modality IN ('pure_infrared', 'pure_visible', 'dual_ir_visible')
    ),
    summary TEXT NOT NULL DEFAULT '',
    lifecycle_status TEXT NOT NULL DEFAULT 'active' CHECK (
        lifecycle_status IN ('draft', 'active', 'retired')
    ),
    primary_source_ref_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(manufacturer_id, model),
    FOREIGN KEY(manufacturer_id) REFERENCES manufacturers(id),
    FOREIGN KEY(primary_source_ref_id) REFERENCES source_references(id)
);

CREATE TABLE IF NOT EXISTS spec_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_key TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    value_type TEXT NOT NULL CHECK (
        value_type IN ('text', 'number', 'boolean', 'json')
    ),
    category TEXT NOT NULL DEFAULT 'general',
    default_unit TEXT,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS product_spec_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    spec_definition_id INTEGER NOT NULL,
    value_text TEXT,
    value_number REAL,
    value_boolean INTEGER CHECK (value_boolean IN (0, 1) OR value_boolean IS NULL),
    value_json TEXT,
    unit TEXT,
    comparator TEXT CHECK (
        comparator IN ('eq', 'lt', 'lte', 'gt', 'gte', 'range') OR comparator IS NULL
    ),
    source_ref_id INTEGER NOT NULL,
    source_excerpt TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(product_id, spec_definition_id),
    FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    FOREIGN KEY(spec_definition_id) REFERENCES spec_definitions(id),
    FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
);

CREATE TABLE IF NOT EXISTS product_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    price_type TEXT NOT NULL CHECK (
        price_type IN ('public_list', 'dealer_reference', 'historical', 'inquiry', 'unknown')
    ),
    amount_min REAL,
    amount_max REAL,
    currency TEXT,
    tax_included INTEGER CHECK (tax_included IN (0, 1) OR tax_included IS NULL),
    effective_from TEXT,
    effective_to TEXT,
    source_ref_id INTEGER NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
);

CREATE TABLE IF NOT EXISTS product_media (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    media_type TEXT NOT NULL CHECK (
        media_type IN ('product_image', 'datasheet_page', 'sample_image', 'other')
    ),
    uri TEXT NOT NULL,
    caption TEXT NOT NULL DEFAULT '',
    source_ref_id INTEGER NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(product_id, uri),
    FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
);

CREATE TABLE IF NOT EXISTS product_scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    scenario TEXT NOT NULL,
    suitability TEXT NOT NULL DEFAULT 'supported' CHECK (
        suitability IN ('supported', 'conditional', 'not_recommended')
    ),
    note TEXT NOT NULL DEFAULT '',
    source_ref_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(product_id, scenario),
    FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
);

CREATE TABLE IF NOT EXISTS fusion_compatibility (
    product_id INTEGER PRIMARY KEY,
    mambadfuse_role TEXT NOT NULL CHECK (
        mambadfuse_role IN ('infrared_source', 'visible_source', 'dual_source_candidate', 'not_applicable')
    ),
    infrared_stream_access INTEGER CHECK (infrared_stream_access IN (0, 1) OR infrared_stream_access IS NULL),
    visible_stream_access INTEGER CHECK (visible_stream_access IN (0, 1) OR visible_stream_access IS NULL),
    independent_modal_streams INTEGER CHECK (independent_modal_streams IN (0, 1) OR independent_modal_streams IS NULL),
    hardware_trigger INTEGER CHECK (hardware_trigger IN (0, 1) OR hardware_trigger IS NULL),
    timestamp_sync INTEGER CHECK (timestamp_sync IN (0, 1) OR timestamp_sync IS NULL),
    sdk_access INTEGER CHECK (sdk_access IN (0, 1) OR sdk_access IS NULL),
    calibration_data INTEGER CHECK (calibration_data IN (0, 1) OR calibration_data IS NULL),
    registration_required INTEGER NOT NULL CHECK (registration_required IN (0, 1)),
    suitability_status TEXT NOT NULL CHECK (
        suitability_status IN ('suitable', 'conditional', 'not_suitable', 'unknown')
    ),
    notes TEXT NOT NULL DEFAULT '',
    source_ref_id INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE,
    FOREIGN KEY(source_ref_id) REFERENCES source_references(id)
);

CREATE INDEX IF NOT EXISTS idx_products_modality_status
    ON products(modality, lifecycle_status);
CREATE INDEX IF NOT EXISTS idx_product_spec_number
    ON product_spec_values(spec_definition_id, value_number);
CREATE INDEX IF NOT EXISTS idx_product_spec_boolean
    ON product_spec_values(spec_definition_id, value_boolean);
CREATE INDEX IF NOT EXISTS idx_product_prices_product
    ON product_prices(product_id, price_type);
CREATE INDEX IF NOT EXISTS idx_source_references_hash
    ON source_references(source_sha256, page_number);
"""

MIGRATION_002 = """
CREATE TABLE IF NOT EXISTS product_spec_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_table_id INTEGER,
    source_document_title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    source_page INTEGER NOT NULL CHECK (source_page > 0),
    source_table_index INTEGER NOT NULL CHECK (source_table_index > 0),
    manufacturer_name TEXT NOT NULL,
    model TEXT NOT NULL,
    product_name TEXT NOT NULL,
    modality TEXT NOT NULL CHECK (
        modality IN ('pure_infrared', 'pure_visible', 'dual_ir_visible')
    ),
    spec_key TEXT NOT NULL,
    spec_label TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    value_type TEXT NOT NULL CHECK (
        value_type IN ('text', 'number', 'boolean', 'json')
    ),
    raw_value TEXT NOT NULL,
    normalized_value_json TEXT NOT NULL,
    unit TEXT,
    comparator TEXT CHECK (
        comparator IN ('eq', 'lt', 'lte', 'gt', 'gte', 'range') OR comparator IS NULL
    ),
    raw_row_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.8 CHECK (confidence >= 0 AND confidence <= 1),
    review_status TEXT NOT NULL DEFAULT 'pending' CHECK (
        review_status IN ('pending', 'approved', 'rejected', 'published')
    ),
    review_note TEXT NOT NULL DEFAULT '',
    reviewed_by TEXT,
    reviewed_at TEXT,
    published_product_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_sha256, source_page, source_table_index, model, spec_key),
    FOREIGN KEY(published_product_id) REFERENCES products(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_product_candidates_status
    ON product_spec_candidates(review_status, model);
CREATE INDEX IF NOT EXISTS idx_product_candidates_source
    ON product_spec_candidates(source_sha256, source_page, source_table_index);
"""


def apply_migrations(conn: Any) -> None:
    if getattr(conn, "backend", "sqlite") == "mysql":
        row = conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None or int(row["version"]) < 3:
            raise RuntimeError("MySQL产品库结构未初始化或版本过低")
        return
    conn.executescript(MIGRATION_001)
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_migrations(version, name, applied_at)
        VALUES (1, 'structured_product_catalog', datetime('now'))
        """
    )
    conn.executescript(MIGRATION_002)
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_migrations(version, name, applied_at)
        VALUES (2, 'product_spec_review_workflow', datetime('now'))
        """
    )


def init_product_schema(db_path: Optional[Path] = None) -> None:
    with connect(db_path) as conn:
        apply_migrations(conn)
        conn.commit()
