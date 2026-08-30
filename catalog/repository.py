from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from catalog.schema import DEFAULT_DB_PATH, apply_migrations, connect, product_database_backend


PROJECT_DIR = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _source_ref_id(conn, source: Dict[str, Any], page_number: int) -> int:
    source_path = Path(source["path"])
    if not source_path.is_absolute():
        source_path = PROJECT_DIR / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"产品来源文档不存在：{source_path}")

    file_hash = sha256_file(source_path)
    now = now_utc()
    conn.execute(
        """
        INSERT INTO source_references(
            document_title, source_path, source_url, source_sha256,
            page_number, retrieved_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_title, source_sha256, page_number) DO UPDATE SET
            source_path = excluded.source_path,
            source_url = COALESCE(excluded.source_url, source_references.source_url),
            retrieved_at = COALESCE(excluded.retrieved_at, source_references.retrieved_at)
        """,
        (
            source["document_title"],
            str(source_path.relative_to(PROJECT_DIR)),
            source.get("url"),
            file_hash,
            int(page_number),
            source.get("retrieved_at"),
            now,
        ),
    )
    row = conn.execute(
        """
        SELECT id FROM source_references
        WHERE document_title = ? AND source_sha256 = ? AND page_number = ?
        """,
        (source["document_title"], file_hash, int(page_number)),
    ).fetchone()
    return int(row["id"])


def import_product_seed(
    seed_path: Path,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """事务化、幂等地把一个人工核验后的产品种子导入结构化目录。"""
    seed_path = Path(seed_path)
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    product = seed["product"]
    manufacturer = seed["manufacturer"]
    source = seed["source"]
    now = now_utc()

    with connect(db_path) as conn:
        apply_migrations(conn)

        conn.execute(
            """
            INSERT INTO manufacturers(name, website, country_code, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                website = excluded.website,
                country_code = excluded.country_code,
                updated_at = excluded.updated_at
            """,
            (
                manufacturer["name"],
                manufacturer.get("website"),
                manufacturer.get("country_code"),
                now,
                now,
            ),
        )
        manufacturer_id = int(
            conn.execute(
                "SELECT id FROM manufacturers WHERE name = ?",
                (manufacturer["name"],),
            ).fetchone()["id"]
        )

        primary_source_ref_id = _source_ref_id(
            conn, source, int(source.get("primary_page", 1))
        )
        conn.execute(
            """
            INSERT INTO products(
                manufacturer_id, model, name, modality, summary,
                lifecycle_status, primary_source_ref_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(manufacturer_id, model) DO UPDATE SET
                name = excluded.name,
                modality = excluded.modality,
                summary = excluded.summary,
                lifecycle_status = excluded.lifecycle_status,
                primary_source_ref_id = excluded.primary_source_ref_id,
                updated_at = excluded.updated_at
            """,
            (
                manufacturer_id,
                product["model"],
                product["name"],
                product["modality"],
                product.get("summary", ""),
                product.get("lifecycle_status", "active"),
                primary_source_ref_id,
                now,
                now,
            ),
        )
        product_id = int(
            conn.execute(
                "SELECT id FROM products WHERE manufacturer_id = ? AND model = ?",
                (manufacturer_id, product["model"]),
            ).fetchone()["id"]
        )

        # 这个种子文件代表该产品当前的完整结构化快照，先删后插可避免旧字段残留。
        for table in [
            "product_spec_values",
            "product_prices",
            "product_media",
            "product_scenarios",
            "fusion_compatibility",
        ]:
            conn.execute(f"DELETE FROM {table} WHERE product_id = ?", (product_id,))

        for spec in seed.get("specs", []):
            definition = spec["definition"]
            conn.execute(
                """
                INSERT INTO spec_definitions(
                    spec_key, label, value_type, category, default_unit,
                    description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(spec_key) DO UPDATE SET
                    label = excluded.label,
                    value_type = excluded.value_type,
                    category = excluded.category,
                    default_unit = excluded.default_unit,
                    description = excluded.description
                """,
                (
                    definition["key"],
                    definition["label"],
                    definition["value_type"],
                    definition.get("category", "general"),
                    definition.get("default_unit"),
                    definition.get("description", ""),
                    now,
                ),
            )
            definition_id = int(
                conn.execute(
                    "SELECT id FROM spec_definitions WHERE spec_key = ?",
                    (definition["key"],),
                ).fetchone()["id"]
            )
            source_ref_id = _source_ref_id(conn, source, int(spec["source_page"]))
            value_type = definition["value_type"]
            value = spec.get("value")
            conn.execute(
                """
                INSERT INTO product_spec_values(
                    product_id, spec_definition_id, value_text, value_number,
                    value_boolean, value_json, unit, comparator, source_ref_id,
                    source_excerpt, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    definition_id,
                    str(value) if value_type == "text" and value is not None else None,
                    float(value) if value_type == "number" and value is not None else None,
                    int(bool(value)) if value_type == "boolean" and value is not None else None,
                    json.dumps(value, ensure_ascii=False) if value_type == "json" else None,
                    spec.get("unit") or definition.get("default_unit"),
                    spec.get("comparator", "eq"),
                    source_ref_id,
                    spec.get("source_excerpt", ""),
                    now,
                    now,
                ),
            )

        for price in seed.get("prices", []):
            source_ref_id = _source_ref_id(conn, source, int(price["source_page"]))
            conn.execute(
                """
                INSERT INTO product_prices(
                    product_id, price_type, amount_min, amount_max, currency,
                    tax_included, effective_from, effective_to, source_ref_id,
                    note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    price["price_type"],
                    price.get("amount_min"),
                    price.get("amount_max"),
                    price.get("currency"),
                    price.get("tax_included"),
                    price.get("effective_from"),
                    price.get("effective_to"),
                    source_ref_id,
                    price.get("note", ""),
                    now,
                    now,
                ),
            )

        for media in seed.get("media", []):
            source_ref_id = _source_ref_id(conn, source, int(media["source_page"]))
            conn.execute(
                """
                INSERT INTO product_media(
                    product_id, media_type, uri, caption,
                    source_ref_id, sort_order, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    media["media_type"],
                    media["uri"],
                    media.get("caption", ""),
                    source_ref_id,
                    int(media.get("sort_order", 0)),
                    now,
                ),
            )

        for scenario in seed.get("scenarios", []):
            source_ref_id = _source_ref_id(conn, source, int(scenario["source_page"]))
            conn.execute(
                """
                INSERT INTO product_scenarios(
                    product_id, scenario, suitability, note,
                    source_ref_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    scenario["scenario"],
                    scenario.get("suitability", "supported"),
                    scenario.get("note", ""),
                    source_ref_id,
                    now,
                ),
            )

        compatibility = seed.get("fusion_compatibility")
        if compatibility:
            source_ref_id = _source_ref_id(
                conn, source, int(compatibility["source_page"])
            )
            conn.execute(
                """
                INSERT INTO fusion_compatibility(
                    product_id, mambadfuse_role, infrared_stream_access,
                    visible_stream_access, independent_modal_streams,
                    hardware_trigger, timestamp_sync, sdk_access,
                    calibration_data, registration_required, suitability_status,
                    notes, source_ref_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id,
                    compatibility["mambadfuse_role"],
                    compatibility.get("infrared_stream_access"),
                    compatibility.get("visible_stream_access"),
                    compatibility.get("independent_modal_streams"),
                    compatibility.get("hardware_trigger"),
                    compatibility.get("timestamp_sync"),
                    compatibility.get("sdk_access"),
                    compatibility.get("calibration_data"),
                    int(bool(compatibility.get("registration_required", True))),
                    compatibility.get("suitability_status", "unknown"),
                    compatibility.get("notes", ""),
                    source_ref_id,
                    now,
                ),
            )

        conn.commit()

    return get_product(product["model"], db_path=db_path)


def apply_product_profile(
    profile: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """补充已发布产品的名称、价格、媒体、场景与融合适配信息。"""
    model = profile["model"]
    now = now_utc()
    with connect(db_path) as conn:
        apply_migrations(conn)
        product = conn.execute(
            """
            SELECT p.id, p.primary_source_ref_id, sr.document_title,
                   sr.source_path, sr.page_number
            FROM products p
            JOIN source_references sr ON p.primary_source_ref_id = sr.id
            WHERE p.model = ?
            """,
            (model,),
        ).fetchone()
        if product is None:
            raise KeyError(f"产品尚未发布：{model}")
        product_id = int(product["id"])
        conn.execute(
            """
            UPDATE products
            SET name = ?, summary = ?, modality = ?, lifecycle_status = 'active', updated_at = ?
            WHERE id = ?
            """,
            (
                profile["name"], profile.get("summary", ""),
                profile["modality"], now, product_id,
            ),
        )
        if profile.get("source_url"):
            conn.execute(
                "UPDATE source_references SET source_url = ? WHERE id = ?",
                (profile["source_url"], product["primary_source_ref_id"]),
            )

        source = {
            "document_title": product["document_title"],
            "path": product["source_path"],
            "url": profile.get("source_url"),
        }
        for table in ["product_prices", "product_media", "product_scenarios", "fusion_compatibility"]:
            conn.execute(f"DELETE FROM {table} WHERE product_id = ?", (product_id,))

        price = profile.get("price")
        if price:
            source_ref_id = _source_ref_id(conn, source, int(price.get("source_page", 1)))
            conn.execute(
                """
                INSERT INTO product_prices(
                    product_id, price_type, amount_min, amount_max, currency,
                    tax_included, source_ref_id, note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id, price.get("price_type", "inquiry"),
                    price.get("amount_min"), price.get("amount_max"),
                    price.get("currency"), price.get("tax_included"),
                    source_ref_id, price.get("note", ""), now, now,
                ),
            )

        for media in profile.get("media", []):
            source_ref_id = _source_ref_id(conn, source, int(media.get("source_page", 1)))
            conn.execute(
                """
                INSERT INTO product_media(
                    product_id, media_type, uri, caption,
                    source_ref_id, sort_order, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id, media.get("media_type", "product_image"), media["uri"],
                    media.get("caption", ""), source_ref_id,
                    int(media.get("sort_order", 0)), now,
                ),
            )

        for scenario in profile.get("scenarios", []):
            source_ref_id = _source_ref_id(conn, source, int(scenario.get("source_page", 1)))
            conn.execute(
                """
                INSERT INTO product_scenarios(
                    product_id, scenario, suitability, note,
                    source_ref_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id, scenario["scenario"], scenario.get("suitability", "supported"),
                    scenario.get("note", ""), source_ref_id, now,
                ),
            )

        compatibility = profile.get("fusion_compatibility")
        if compatibility:
            source_ref_id = _source_ref_id(
                conn, source, int(compatibility.get("source_page", 1))
            )
            conn.execute(
                """
                INSERT INTO fusion_compatibility(
                    product_id, mambadfuse_role, infrared_stream_access,
                    visible_stream_access, independent_modal_streams,
                    hardware_trigger, timestamp_sync, sdk_access,
                    calibration_data, registration_required, suitability_status,
                    notes, source_ref_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product_id, compatibility["mambadfuse_role"],
                    compatibility.get("infrared_stream_access"),
                    compatibility.get("visible_stream_access"),
                    compatibility.get("independent_modal_streams"),
                    compatibility.get("hardware_trigger"),
                    compatibility.get("timestamp_sync"),
                    compatibility.get("sdk_access"),
                    compatibility.get("calibration_data"),
                    int(bool(compatibility.get("registration_required", True))),
                    compatibility.get("suitability_status", "unknown"),
                    compatibility.get("notes", ""), source_ref_id, now,
                ),
            )
        conn.commit()
    return get_product(model, db_path=db_path)


def _typed_value(row: Dict[str, Any]) -> Any:
    value_type = row["value_type"]
    if value_type == "number":
        return row["value_number"]
    if value_type == "boolean":
        value = row["value_boolean"]
        return None if value is None else bool(value)
    if value_type == "json":
        return json.loads(row["value_json"]) if row["value_json"] else None
    return row["value_text"]


def get_product(model: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    with connect(db_path) as conn:
        apply_migrations(conn)
        product_row = conn.execute(
            """
            SELECT p.*, m.name AS manufacturer_name, m.website AS manufacturer_website,
                   sr.document_title, sr.source_path, sr.source_url,
                   sr.source_sha256, sr.page_number AS primary_source_page
            FROM products p
            JOIN manufacturers m ON p.manufacturer_id = m.id
            LEFT JOIN source_references sr ON p.primary_source_ref_id = sr.id
            WHERE p.model = ?
            """,
            (model,),
        ).fetchone()
        if product_row is None:
            return {}
        product = dict(product_row)
        product_id = int(product["id"])

        specs = []
        for row in conn.execute(
            """
            SELECT sd.spec_key, sd.label, sd.value_type, sd.category,
                   psv.value_text, psv.value_number, psv.value_boolean,
                   psv.value_json, psv.unit, psv.comparator,
                   psv.source_excerpt, sr.document_title,
                   sr.page_number AS source_page
            FROM product_spec_values psv
            JOIN spec_definitions sd ON psv.spec_definition_id = sd.id
            JOIN source_references sr ON psv.source_ref_id = sr.id
            WHERE psv.product_id = ?
            ORDER BY sd.category, sd.id
            """,
            (product_id,),
        ).fetchall():
            item = dict(row)
            item["value"] = _typed_value(item)
            specs.append(item)

        prices = [
            dict(row)
            for row in conn.execute(
                """
                SELECT pp.*, sr.document_title, sr.page_number AS source_page
                FROM product_prices pp
                JOIN source_references sr ON pp.source_ref_id = sr.id
                WHERE pp.product_id = ? ORDER BY pp.id DESC
                """,
                (product_id,),
            ).fetchall()
        ]
        media = [
            dict(row)
            for row in conn.execute(
                """
                SELECT pm.*, sr.document_title, sr.page_number AS source_page
                FROM product_media pm
                JOIN source_references sr ON pm.source_ref_id = sr.id
                WHERE pm.product_id = ? ORDER BY pm.sort_order, pm.id
                """,
                (product_id,),
            ).fetchall()
        ]
        scenarios = [
            dict(row)
            for row in conn.execute(
                """
                SELECT ps.*, sr.document_title, sr.page_number AS source_page
                FROM product_scenarios ps
                JOIN source_references sr ON ps.source_ref_id = sr.id
                WHERE ps.product_id = ? ORDER BY ps.id
                """,
                (product_id,),
            ).fetchall()
        ]
        compatibility_row = conn.execute(
            """
            SELECT fc.*, sr.document_title, sr.page_number AS source_page
            FROM fusion_compatibility fc
            JOIN source_references sr ON fc.source_ref_id = sr.id
            WHERE fc.product_id = ?
            """,
            (product_id,),
        ).fetchone()

    product["specs"] = specs
    product["prices"] = prices
    product["media"] = media
    product["scenarios"] = scenarios
    product["fusion_compatibility"] = dict(compatibility_row) if compatibility_row else None
    return product


def list_products(
    modality: Optional[str] = None,
    include_retired: bool = False,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    with connect(db_path) as conn:
        apply_migrations(conn)
        conditions = []
        params: List[Any] = []
        if modality:
            conditions.append("p.modality = ?")
            params.append(modality)
        if not include_retired:
            conditions.append("p.lifecycle_status != 'retired'")
        where_sql = "WHERE " + " AND ".join(conditions) if conditions else ""
        rows = conn.execute(
            f"""
            SELECT p.id, p.model, p.name, p.modality, p.summary,
                   p.lifecycle_status, p.updated_at,
                   m.name AS manufacturer_name
            FROM products p JOIN manufacturers m ON p.manufacturer_id = m.id
            {where_sql}
            ORDER BY m.name, p.model
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def search_products(
    modality: Optional[str] = None,
    min_width: Optional[float] = None,
    min_height: Optional[float] = None,
    min_fps: Optional[float] = None,
    sdk_required: Optional[bool] = None,
    model: Optional[str] = None,
    required_protocols: Optional[List[str]] = None,
    min_ip_rating: Optional[int] = None,
    fusion_required: Optional[bool] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    currency: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """使用确定的结构化字段过滤产品，不依赖大模型猜测参数。"""
    conditions = ["p.lifecycle_status = 'active'"]
    params: List[Any] = []
    if modality:
        conditions.append("p.modality = ?")
        params.append(modality)
    if model:
        conditions.append("UPPER(p.model) = UPPER(?)")
        params.append(model)

    for spec_key, threshold in [
        ("resolution_width", min_width),
        ("resolution_height", min_height),
        ("stream_frame_rate", min_fps),
    ]:
        if threshold is None:
            continue
        conditions.append(
            """
            EXISTS (
                SELECT 1
                FROM product_spec_values psv
                JOIN spec_definitions sd ON psv.spec_definition_id = sd.id
                WHERE psv.product_id = p.id
                  AND sd.spec_key = ?
                  AND psv.value_number >= ?
            )
            """
        )
        params.extend([spec_key, float(threshold)])

    if sdk_required is not None:
        conditions.append(
            """
            EXISTS (
                SELECT 1
                FROM product_spec_values psv
                JOIN spec_definitions sd ON psv.spec_definition_id = sd.id
                WHERE psv.product_id = p.id
                  AND sd.spec_key = 'sdk_api_support'
                  AND psv.value_boolean = ?
            )
            """
        )
        params.append(int(sdk_required))

    for protocol in required_protocols or []:
        conditions.append(
            """
            EXISTS (
                SELECT 1
                FROM product_spec_values psv
                JOIN spec_definitions sd ON psv.spec_definition_id = sd.id
                WHERE psv.product_id = p.id
                  AND sd.spec_key IN ('network_protocols', 'interface_standards', 'data_interface')
                  AND UPPER(COALESCE(psv.value_text, psv.value_json, '')) LIKE UPPER(?)
            )
            """
        )
        params.append(f"%{protocol}%")

    if min_ip_rating is not None:
        conditions.append(
            """
            EXISTS (
                SELECT 1
                FROM product_spec_values psv
                JOIN spec_definitions sd ON psv.spec_definition_id = sd.id
                WHERE psv.product_id = p.id
                  AND sd.spec_key = 'ip_rating'
                  AND CAST(REPLACE(UPPER(psv.value_text), 'IP', '') AS INTEGER) >= ?
            )
            """
        )
        params.append(int(min_ip_rating))

    if fusion_required:
        conditions.append(
            """
            EXISTS (
                SELECT 1 FROM fusion_compatibility fc
                WHERE fc.product_id = p.id
                  AND fc.mambadfuse_role != 'not_applicable'
                  AND fc.suitability_status IN ('suitable', 'conditional')
            )
            """
        )

    if min_price is not None or max_price is not None:
        price_conditions = [
            "pp.product_id = p.id",
            "pp.amount_min IS NOT NULL",
        ]
        if min_price is not None:
            # 产品公开价格区间必须与用户下限有交集。
            price_conditions.append("COALESCE(pp.amount_max, pp.amount_min) >= ?")
            params.append(float(min_price))
        if max_price is not None:
            # 产品公开价格区间必须与用户上限有交集。
            price_conditions.append("COALESCE(pp.amount_min, pp.amount_max) <= ?")
            params.append(float(max_price))
        if currency:
            price_conditions.append("UPPER(pp.currency) = UPPER(?)")
            params.append(currency)
        conditions.append(
            "EXISTS (SELECT 1 FROM product_prices pp WHERE "
            + " AND ".join(price_conditions)
            + ")"
        )

    with connect(db_path) as conn:
        apply_migrations(conn)
        rows = conn.execute(
            f"""
            SELECT p.id, p.model, p.name, p.modality, p.summary,
                   p.lifecycle_status, p.updated_at,
                   m.name AS manufacturer_name
            FROM products p JOIN manufacturers m ON p.manufacturer_id = m.id
            WHERE {' AND '.join(conditions)}
            ORDER BY m.name, p.model
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_catalog_status(db_path: Optional[Path] = None) -> Dict[str, Any]:
    with connect(db_path) as conn:
        apply_migrations(conn)
        return {
            "database_backend": "sqlite" if db_path is not None else product_database_backend(),
            "product_count": conn.execute("SELECT COUNT(*) FROM products").fetchone()[0],
            "manufacturer_count": conn.execute("SELECT COUNT(*) FROM manufacturers").fetchone()[0],
            "spec_value_count": conn.execute("SELECT COUNT(*) FROM product_spec_values").fetchone()[0],
            "media_count": conn.execute("SELECT COUNT(*) FROM product_media").fetchone()[0],
            "price_record_count": conn.execute("SELECT COUNT(*) FROM product_prices").fetchone()[0],
            "migration_versions": [
                row[0]
                for row in conn.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                ).fetchall()
            ],
        }


MODALITY_LABELS = {
    "pure_infrared": "纯红外设备",
    "pure_visible": "纯可见光设备",
    "dual_ir_visible": "红外可见光双模态设备",
}


def list_product_cards(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    cards = []
    for row in list_products(db_path=db_path):
        detail = get_product(row["model"], db_path=db_path)
        spec_map = {item["spec_key"]: item for item in detail["specs"]}
        price = detail["prices"][0] if detail["prices"] else None
        if price and price["price_type"] == "inquiry":
            price_display = "公开资料未标价，需询价"
        elif price and price["amount_min"] is not None:
            maximum = price["amount_max"] if price["amount_max"] is not None else price["amount_min"]
            price_display = f"{price.get('currency') or ''} {price['amount_min']:g} - {maximum:g}"
        else:
            price_display = "价格未知"

        highlights = []
        for key in [
            "infrared_resolution",
            "stream_frame_rate",
            "spectral_band",
            "sdk_api_support",
            "operating_temperature",
        ]:
            item = spec_map.get(key)
            if not item:
                continue
            value = item["value"]
            if isinstance(value, bool):
                value = "支持" if value else "不支持"
            elif isinstance(value, list):
                separator = "~" if item.get("comparator") == "range" and len(value) == 2 else "、"
                value = separator.join(str(v) for v in value)
            unit = item.get("unit") or ""
            highlights.append(f"{item['label']}：{value}{unit}")

        compatibility = detail.get("fusion_compatibility") or {}
        limitations = [compatibility.get("notes", "")] if compatibility.get("notes") else []
        image_url = detail["media"][0]["uri"] if detail["media"] else ""
        cards.append(
            {
                "id": detail["id"],
                "model": detail["model"],
                "name": detail["name"],
                "manufacturer": detail["manufacturer_name"],
                "category": MODALITY_LABELS.get(detail["modality"], detail["modality"]),
                "description": detail["summary"],
                "tags": [detail["model"], MODALITY_LABELS.get(detail["modality"], detail["modality"])],
                "highlights": highlights,
                "limitations": limitations,
                "price_display": price_display,
                "image_url": image_url,
                "source_document": detail.get("document_title", ""),
                "source_page": detail.get("primary_source_page"),
                "specs": detail["specs"],
            }
        )
    return cards
