from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_PATH = PROJECT_DIR / "storage" / "kb.sqlite3"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "deploy" / "mysql" / "init" / "002_product_data.sql"

TABLES = [
    "manufacturers",
    "source_references",
    "products",
    "spec_definitions",
    "product_spec_values",
    "product_prices",
    "product_media",
    "product_scenarios",
    "fusion_compatibility",
    "product_spec_candidates",
]
IDENTITY_TABLES = set(TABLES) - {"fusion_compatibility"}
BOOLEAN_COLUMNS = {
    "product_spec_values": {"value_boolean"},
    "product_prices": {"tax_included"},
    "fusion_compatibility": {
        "infrared_stream_access", "visible_stream_access", "independent_modal_streams",
        "hardware_trigger", "timestamp_sync", "sdk_access", "calibration_data",
        "registration_required",
    },
}
JSON_COLUMNS = {
    "product_spec_values": {"value_json"},
    "product_spec_candidates": {"normalized_value_json", "raw_row_json"},
}
TIMESTAMP_COLUMNS = {"created_at", "updated_at", "retrieved_at", "reviewed_at"}
DATE_COLUMNS = {"effective_from", "effective_to"}


def mysql_utf8_literal(value: str) -> str:
    if value == "":
        return "_utf8mb4''"
    return f"CONVERT(0x{value.encode('utf-8').hex()} USING utf8mb4)"


def normalize_timestamp(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed.strftime("%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return text.replace("T", " ")


def sql_value(table: str, column: str, value: Any) -> str:
    if value is None:
        return "NULL"
    if column in TIMESTAMP_COLUMNS:
        normalized = normalize_timestamp(value)
        return "NULL" if normalized is None else mysql_utf8_literal(normalized)
    if column in DATE_COLUMNS:
        return "NULL" if value == "" else mysql_utf8_literal(str(value)[:10])
    if column in BOOLEAN_COLUMNS.get(table, set()):
        return "1" if bool(value) else "0"
    if column in JSON_COLUMNS.get(table, set()):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {"raw": value}
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return f"CAST({mysql_utf8_literal(payload)} AS JSON)"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    return mysql_utf8_literal(str(value))


def export(sqlite_path: Path, output_path: Path) -> dict[str, Any]:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite数据库不存在：{sqlite_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    counts: dict[str, int] = {}
    lines = [
        "SET NAMES utf8mb4;",
        "SET time_zone = '+08:00';",
        "SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;",
        "START TRANSACTION;",
    ]
    try:
        for table in TABLES:
            columns = [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')]
            rows = conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            counts[table] = len(rows)
            if not rows:
                continue
            column_sql = ", ".join(f'`{column}`' for column in columns)
            for row in rows:
                values_sql = ", ".join(
                    sql_value(table, column, row[column]) for column in columns
                )
                lines.append(f"INSERT INTO `{table}` ({column_sql}) VALUES ({values_sql});")

        lines.append("COMMIT;")
        for table in sorted(IDENTITY_TABLES):
            maximum = conn.execute(f'SELECT COALESCE(MAX(id), 0) FROM "{table}"').fetchone()[0]
            lines.append(f"ALTER TABLE `{table}` AUTO_INCREMENT = {int(maximum) + 1};")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {
            "sqlite_path": str(sqlite_path),
            "output_path": str(output_path),
            "tables": len(TABLES),
            "total_rows": sum(counts.values()),
            "counts": counts,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出SQLite产品表为MySQL事务脚本")
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    print(json.dumps(export(args.sqlite, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
