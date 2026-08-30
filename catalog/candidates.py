from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from catalog.repository import _source_ref_id, now_utc, sha256_file
from catalog.schema import DEFAULT_DB_PATH, apply_migrations, connect


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODALITY_NAMES = {
    "纯红外设备": "pure_infrared",
    "纯可见光设备": "pure_visible",
    "红外可见光双模态设备": "dual_ir_visible",
}

CATEGORY_LABELS = {
    "热成像参数": "imaging",
    "测温参数": "measurement",
    "可见光参数": "imaging",
    "性能": "imaging",
    "图像参数": "stream",
    "协议和存储": "integration",
    "系统功能": "system",
    "硬件接口": "interface",
    "电气特性": "interface",
    "环境参数": "environment",
    "结构": "physical",
    "物理参数": "physical",
    "一般规范": "general",
}

SPEC_RULES: Dict[str, Tuple[str, str, str, Optional[str]]] = {
    "探测器类型": ("detector_type", "text", "imaging", None),
    "传感器类型": ("sensor_type", "text", "imaging", None),
    "传感器型号": ("sensor_model", "text", "imaging", None),
    "红外分辨率": ("infrared_resolution", "text", "imaging", None),
    "最大分辨率": ("visible_resolution", "text", "imaging", None),
    "分辨率": ("visible_resolution", "text", "imaging", None),
    "热灵敏度": ("thermal_sensitivity", "text", "imaging", None),
    "热成像镜头": ("infrared_lens", "text", "optics", None),
    "可见光镜头": ("visible_lens", "text", "optics", None),
    "镜头接口": ("lens_mount", "text", "optics", None),
    "对焦方式": ("focus_method", "text", "optics", None),
    "像元尺寸": ("pixel_pitch", "text", "imaging", None),
    "最大帧率": ("stream_frame_rate", "number", "stream", "fps"),
    "帧率": ("stream_frame_rate", "number", "stream", "fps"),
    "视频压缩标准": ("video_codec", "text", "stream", None),
    "图片编码格式": ("image_encoding", "text", "stream", None),
    "像素格式": ("pixel_format", "text", "stream", None),
    "码流": ("stream_profile", "text", "stream", None),
    "网络协议": ("network_protocols", "json", "integration", None),
    "协议/标准": ("interface_standards", "json", "integration", None),
    "SDK/API": ("sdk_api_support", "boolean", "integration", None),
    "数据接口": ("data_interface", "text", "interface", None),
    "工作温度": ("operating_temperature", "text", "environment", None),
    "温度": ("operating_temperature", "text", "environment", None),
    "IP防护等级": ("ip_rating", "text", "environment", None),
    "防护等级": ("ip_rating", "text", "environment", None),
    "功耗": ("power_consumption", "text", "physical", None),
    "典型功耗": ("power_consumption", "text", "physical", None),
    "尺寸": ("dimensions", "text", "physical", None),
    "外形尺寸": ("dimensions", "text", "physical", None),
    "净重": ("weight", "text", "physical", None),
    "重量": ("weight", "text", "physical", None),
    "测温范围": ("temperature_measurement_range", "text", "measurement", None),
    "测温精度": ("temperature_accuracy", "text", "measurement", None),
    "伪彩": ("palette_modes", "json", "imaging", None),
    "调色板": ("palette_modes", "json", "imaging", None),
    "特色功能": ("infrared_features", "json", "imaging", None),
    "热成像功能": ("infrared_features", "json", "imaging", None),
    "测温对象设置": ("measurement_targets", "text", "measurement", None),
    "测温功能": ("measurement_features", "json", "measurement", None),
    "本地存储": ("local_storage", "text", "integration", None),
    "语言版本": ("languages", "json", "system", None),
    "浏览器": ("browser_support", "boolean", "system", None),
    "用户管理": ("user_management", "text", "system", None),
    "故障检测": ("fault_detection", "json", "system", None),
    "电源接口": ("power_input", "text", "interface", None),
    "网络接口": ("network_interface", "text", "interface", None),
    "报警接口": ("alarm_interface", "text", "interface", None),
    "其他接口": ("serial_interface", "text", "interface", None),
    "工作湿度": ("operating_humidity", "text", "environment", None),
    "湿度": ("operating_humidity", "text", "environment", None),
    "认证": ("certifications", "json", "general", None),
    "靶面尺寸": ("sensor_format", "text", "imaging", None),
    "动态范围": ("dynamic_range", "text", "imaging", None),
    "信噪比": ("signal_to_noise_ratio", "text", "imaging", None),
    "增益": ("gain_range", "text", "imaging", None),
    "曝光时间": ("exposure_time", "text", "imaging", None),
    "快门模式": ("exposure_modes", "json", "imaging", None),
    "黑白/彩色": ("color_mode", "text", "imaging", None),
    "Binning": ("binning_modes", "json", "imaging", None),
    "下采样": ("decimation_modes", "json", "imaging", None),
    "镜像": ("mirror_modes", "json", "imaging", None),
    "数字I/O": ("digital_io", "text", "interface", None),
    "供电": ("power_supply", "text", "interface", None),
    "软件": ("supported_software", "json", "general", None),
    "操作系统": ("operating_systems", "json", "general", None),
    "最低照度": ("minimum_illumination", "text", "imaging", None),
    "补光灯": ("illuminators", "json", "imaging", None),
    "可见光功能": ("visible_features", "json", "imaging", None),
    "智能功能": ("analytics_features", "json", "system", None),
    "安装方式": ("mounting_method", "text", "physical", None),
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _normalized_label(value: str) -> str:
    return _clean(value).replace(" ", "")


def _fallback_spec_key(label: str) -> str:
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:10]
    return f"source_{digest}"


def _infer_source(document_title: str, source_path: str) -> Tuple[str, str]:
    text = f"{document_title} {source_path}"
    if "HIKROBOT" in text.upper() or "海康" in text:
        manufacturer = "杭州海康机器人股份有限公司"
    elif "GUIDEIR" in text.upper() or "高德" in text:
        manufacturer = "武汉高德智感科技有限公司"
    else:
        manufacturer = "待确认厂商"

    modality = "dual_ir_visible" if "双模态" in text else (
        "pure_visible" if "可见光" in text and "红外" not in text else "pure_infrared"
    )
    return manufacturer, modality


def _model_columns(rows: Sequence[Sequence[Any]]) -> Tuple[List[str], List[int], int]:
    header_labels = {"型号", "产品型号", "技术指标", "技术参数", "规格参数", "参数名称"}
    for row_index, raw_row in enumerate(rows[:8]):
        row = [_clean(cell) for cell in raw_row]
        label_index = next(
            (i for i, cell in enumerate(row) if _normalized_label(cell) in header_labels),
            None,
        )
        if label_index is None:
            continue
        models: List[str] = []
        columns: List[int] = []
        for index in range(label_index + 1, len(row)):
            cell = row[index]
            if 3 <= len(cell) <= 60 and re.search(r"[A-Za-z]", cell) and re.search(r"\d", cell):
                models.append(cell.replace(" ", ""))
                columns.append(index)
        if models:
            return models, columns, row_index
    return [], [], -1


def _table_modality(rows: Sequence[Sequence[Any]]) -> Optional[str]:
    text = " ".join(_clean(cell) for row in rows for cell in row)
    has_infrared = any(keyword in text for keyword in ("热成像", "红外", "热像"))
    has_visible = "可见光" in text
    if has_infrared and has_visible:
        return "dual_ir_visible"
    if has_visible:
        return "pure_visible"
    if has_infrared:
        return "pure_infrared"
    return None


def _row_values(row: Sequence[str], model_columns: Sequence[int], label_index: int) -> List[str]:
    if len(model_columns) == 1:
        return [next((cell for cell in row[label_index + 1:] if cell), "")]

    values: List[str] = []
    for model_index, column in enumerate(model_columns):
        lower = label_index + 1 if model_index == 0 else (model_columns[model_index - 1] + column) // 2 + 1
        upper = len(row) - 1 if model_index == len(model_columns) - 1 else (column + model_columns[model_index + 1]) // 2
        segment = [cell for cell in row[lower:upper + 1] if cell]
        values.append(" ".join(segment))
    shared = next((value for value in values if value and value != "/"), "")
    return [value or shared for value in values]


def _normalize_value(value_type: str, raw_value: str) -> Any:
    if value_type == "number":
        match = re.search(r"[-+]?\d+(?:\.\d+)?", raw_value)
        return float(match.group()) if match else raw_value
    if value_type == "boolean":
        if "不支持" in raw_value:
            return False
        return True if "支持" in raw_value else None
    if value_type == "json":
        return [part.strip() for part in re.split(r"[,，;；]", raw_value) if part.strip()]
    return raw_value


def _candidate_rows(table: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    rows = table["rows"]
    models, model_columns, header_index = _model_columns(rows)
    if not models:
        return []
    current_category = "general"
    current_section = ""
    candidates: List[Dict[str, Any]] = []
    for raw_row in rows[header_index + 1:]:
        row = [_clean(cell) for cell in raw_row]
        first_model_boundary = max(1, (model_columns[0] + 1) // 2)
        label_cells = [cell for cell in row[:first_model_boundary] if cell]
        if not label_cells:
            continue
        label = _normalized_label(label_cells[0])
        if label in {"产品型号", "型号", "参数"}:
            continue
        values = _row_values(row, model_columns, max(i for i, cell in enumerate(row[:first_model_boundary]) if cell))
        if label in CATEGORY_LABELS and not any(value and value != "/" for value in values):
            current_category = CATEGORY_LABELS[label]
            current_section = label
            continue
        rule = SPEC_RULES.get(label)
        if label == "传感器分辨率":
            if current_section == "热成像参数":
                rule = ("infrared_resolution", "text", "imaging", None)
            elif current_section == "可见光参数":
                rule = ("visible_resolution", "text", "imaging", None)
        spec_key, value_type, category, unit = rule or (
            _fallback_spec_key(label), "text", current_category, None
        )
        for model, raw_value in zip(models, values):
            if not raw_value or raw_value == "/":
                continue
            normalized_value = _normalize_value(value_type, raw_value)
            candidates.append({
                "model": model,
                "spec_key": spec_key,
                "spec_label": label,
                "category": category,
                "value_type": value_type,
                "raw_value": raw_value,
                "normalized_value": normalized_value,
                "unit": unit,
                "comparator": "eq",
                "raw_row": list(raw_row),
                "confidence": 0.9 if rule else 0.65,
            })
            resolution = re.search(r"(\d{2,5})\s*[×xX]\s*(\d{2,5})", raw_value)
            if resolution and "分辨率" in label:
                for suffix, derived_label, number in [
                    ("width", "分辨率宽度", resolution.group(1)),
                    ("height", "分辨率高度", resolution.group(2)),
                ]:
                    prefix = "infrared_resolution" if spec_key == "infrared_resolution" else "resolution"
                    candidates.append({
                        "model": model,
                        "spec_key": f"{prefix}_{suffix}",
                        "spec_label": derived_label,
                        "category": "imaging",
                        "value_type": "number",
                        "raw_value": raw_value,
                        "normalized_value": float(number),
                        "unit": "px",
                        "comparator": "eq",
                        "raw_row": list(raw_row),
                        "confidence": 0.95,
                    })
    return candidates


def generate_candidates_from_document_tables(db_path: Optional[Path] = None) -> Dict[str, Any]:
    # 文档解析表仍在RAG SQLite；候选产品参数写入已配置的产品数据库。
    source_db_path = Path(db_path or DEFAULT_DB_PATH)
    with connect(source_db_path) as source_conn:
        apply_migrations(source_conn)
        document_columns = {
            row[1] for row in source_conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        has_generations = source_conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'kb_generations'"
        ).fetchone() is not None
        generation_join = (
            "JOIN kb_generations kg ON kg.id = d.generation_id AND kg.status = 'active'"
            if "generation_id" in document_columns and has_generations else ""
        )
        tables = source_conn.execute(
            f"""
            SELECT dt.id, dt.page_number, dt.table_index, dt.rows_json,
                   d.title, d.source, d.category
            FROM document_tables dt
            JOIN documents d ON d.id = dt.doc_id
            {generation_join}
            ORDER BY d.id, dt.page_number, dt.table_index
            """
        ).fetchall()
    generated = 0
    models = set()
    with connect(db_path) as conn:
        apply_migrations(conn)
        for table_row in tables:
            source_path = Path(table_row["source"])
            if not source_path.is_absolute():
                source_path = PROJECT_DIR / source_path
            if not source_path.exists():
                continue
            source_hash = sha256_file(source_path)
            manufacturer, inferred_modality = _infer_source(table_row["title"], str(source_path))
            table = {"rows": json.loads(table_row["rows_json"])}
            detected_modality = _table_modality(table["rows"])
            modality = detected_modality or MODALITY_NAMES.get(table_row["category"], inferred_modality)
            for candidate in _candidate_rows(table):
                model = candidate["model"]
                models.add(model)
                now = now_utc()
                candidate_modality = modality
                if modality == "dual_ir_visible" and detected_modality != "dual_ir_visible" and "FT" not in model.upper():
                    candidate_modality = "pure_visible"
                conn.execute(
                    """
                    INSERT INTO product_spec_candidates(
                        document_table_id, source_document_title, source_path,
                        source_sha256, source_page, source_table_index,
                        manufacturer_name, model, product_name, modality,
                        spec_key, spec_label, category, value_type, raw_value,
                        normalized_value_json, unit, comparator, raw_row_json,
                        confidence, review_note, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_sha256, source_page, source_table_index, model, spec_key)
                    DO UPDATE SET
                        raw_value = excluded.raw_value,
                        raw_row_json = excluded.raw_row_json,
                        confidence = excluded.confidence,
                        updated_at = excluded.updated_at
                    """,
                    (
                        table_row["id"], table_row["title"], str(source_path.relative_to(PROJECT_DIR)),
                        source_hash, table_row["page_number"], table_row["table_index"],
                        manufacturer, model, f"{model} 成像设备", candidate_modality,
                        candidate["spec_key"], candidate["spec_label"], candidate["category"],
                        candidate["value_type"], candidate["raw_value"],
                        json.dumps(candidate["normalized_value"], ensure_ascii=False),
                        candidate["unit"], candidate["comparator"],
                        json.dumps(candidate["raw_row"], ensure_ascii=False),
                        candidate["confidence"], "", now, now,
                    ),
                )
                generated += 1
        conn.commit()
    return {"tables_seen": len(tables), "candidates_seen": generated, "models": sorted(models)}


def list_candidates(
    status: Optional[str] = None,
    model: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    conditions: List[str] = []
    params: List[Any] = []
    if status:
        conditions.append("review_status = ?")
        params.append(status)
    if model:
        conditions.append("model = ?")
        params.append(model)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    with connect(db_path) as conn:
        apply_migrations(conn)
        rows = conn.execute(
            f"SELECT * FROM product_spec_candidates {where} ORDER BY model, source_page, id",
            params,
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["value"] = json.loads(item.pop("normalized_value_json"))
        item["raw_row"] = json.loads(item.pop("raw_row_json"))
        result.append(item)
    return result


def review_candidate(
    candidate_id: int,
    review_status: str,
    updates: Optional[Dict[str, Any]] = None,
    reviewed_by: str = "local_admin",
    review_note: str = "",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if review_status not in {"approved", "rejected", "pending"}:
        raise ValueError("审核状态必须是 pending、approved 或 rejected")
    updates = updates or {}
    spec_key = updates.get("spec_key")
    if spec_key is not None and not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", spec_key):
        raise ValueError("字段键只能使用小写字母、数字和下划线，并以字母开头")
    value_type = updates.get("value_type")
    if value_type is not None and value_type not in {"text", "number", "boolean", "json"}:
        raise ValueError("参数类型不合法")
    comparator = updates.get("comparator")
    if comparator is not None and comparator not in {"eq", "lt", "lte", "gt", "gte", "range"}:
        raise ValueError("比较方式不合法")
    if value_type == "number" and "value" in updates:
        try:
            updates["value"] = float(updates["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError("number类型必须填写数字") from exc
    if value_type == "boolean" and "value" in updates and not isinstance(updates["value"], bool):
        raise ValueError("boolean类型必须是true或false")
    allowed = {"spec_key", "spec_label", "category", "value_type", "unit", "comparator"}
    assignments = ["review_status = ?", "review_note = ?", "reviewed_by = ?", "reviewed_at = ?", "updated_at = ?"]
    now = now_utc()
    values: List[Any] = [review_status, review_note, reviewed_by, now, now]
    for key in allowed:
        if key in updates and updates[key] is not None:
            assignments.append(f"{key} = ?")
            values.append(updates[key])
    if "value" in updates:
        assignments.append("normalized_value_json = ?")
        values.append(json.dumps(updates["value"], ensure_ascii=False))
    values.append(int(candidate_id))
    with connect(db_path) as conn:
        apply_migrations(conn)
        current = conn.execute(
            "SELECT review_status FROM product_spec_candidates WHERE id = ?",
            (int(candidate_id),),
        ).fetchone()
        if current is None:
            raise KeyError(f"候选参数不存在：{candidate_id}")
        if current["review_status"] == "published":
            raise ValueError("已发布参数不能直接修改，请创建新候选版本")
        cursor = conn.execute(
            f"UPDATE product_spec_candidates SET {', '.join(assignments)} WHERE id = ?",
            values,
        )
        conn.commit()
    return next(item for item in list_candidates(db_path=db_path) if item["id"] == int(candidate_id))


def _typed_columns(value_type: str, value: Any) -> Tuple[Any, Any, Any, Any]:
    return (
        str(value) if value_type == "text" and value is not None else None,
        float(value) if value_type == "number" and value is not None else None,
        int(bool(value)) if value_type == "boolean" and value is not None else None,
        json.dumps(value, ensure_ascii=False) if value_type == "json" else None,
    )


def publish_approved_candidates(model: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    with connect(db_path) as conn:
        apply_migrations(conn)
        candidates = conn.execute(
            "SELECT * FROM product_spec_candidates WHERE model = ? AND review_status = 'approved' ORDER BY id",
            (model,),
        ).fetchall()
        if not candidates:
            raise ValueError(f"型号 {model} 没有已通过审核的候选参数")
        first = candidates[0]
        now = now_utc()
        conn.execute(
            """
            INSERT INTO manufacturers(name, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (first["manufacturer_name"], now, now),
        )
        manufacturer_id = int(conn.execute(
            "SELECT id FROM manufacturers WHERE name = ?", (first["manufacturer_name"],)
        ).fetchone()["id"])
        primary_source_ref_id = _source_ref_id(conn, {
            "document_title": first["source_document_title"],
            "path": first["source_path"],
        }, int(first["source_page"]))
        conn.execute(
            """
            INSERT INTO products(
                manufacturer_id, model, name, modality, summary,
                lifecycle_status, primary_source_ref_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, '', 'active', ?, ?, ?)
            ON CONFLICT(manufacturer_id, model) DO UPDATE SET
                primary_source_ref_id = excluded.primary_source_ref_id,
                updated_at = excluded.updated_at
            """,
            (
                manufacturer_id, model, first["product_name"], first["modality"],
                primary_source_ref_id, now, now,
            ),
        )
        product_id = int(conn.execute(
            "SELECT id FROM products WHERE manufacturer_id = ? AND model = ?",
            (manufacturer_id, model),
        ).fetchone()["id"])
        for candidate in candidates:
            source_ref_id = _source_ref_id(conn, {
                "document_title": candidate["source_document_title"],
                "path": candidate["source_path"],
            }, int(candidate["source_page"]))
            conn.execute(
                """
                INSERT INTO spec_definitions(
                    spec_key, label, value_type, category, default_unit,
                    description, created_at
                ) VALUES (?, ?, ?, ?, ?, '', ?)
                ON CONFLICT(spec_key) DO UPDATE SET
                    label = excluded.label,
                    value_type = excluded.value_type,
                    category = excluded.category,
                    default_unit = excluded.default_unit
                """,
                (
                    candidate["spec_key"], candidate["spec_label"], candidate["value_type"],
                    candidate["category"], candidate["unit"], now,
                ),
            )
            definition_id = int(conn.execute(
                "SELECT id FROM spec_definitions WHERE spec_key = ?", (candidate["spec_key"],)
            ).fetchone()["id"])
            value = json.loads(candidate["normalized_value_json"])
            value_text, value_number, value_boolean, value_json = _typed_columns(candidate["value_type"], value)
            conn.execute(
                """
                INSERT INTO product_spec_values(
                    product_id, spec_definition_id, value_text, value_number,
                    value_boolean, value_json, unit, comparator, source_ref_id,
                    source_excerpt, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_id, spec_definition_id) DO UPDATE SET
                    value_text = excluded.value_text,
                    value_number = excluded.value_number,
                    value_boolean = excluded.value_boolean,
                    value_json = excluded.value_json,
                    unit = excluded.unit,
                    comparator = excluded.comparator,
                    source_ref_id = excluded.source_ref_id,
                    source_excerpt = excluded.source_excerpt,
                    updated_at = excluded.updated_at
                """,
                (
                    product_id, definition_id, value_text, value_number, value_boolean,
                    value_json, candidate["unit"], candidate["comparator"], source_ref_id,
                    candidate["raw_value"], now, now,
                ),
            )
            conn.execute(
                """
                UPDATE product_spec_candidates
                SET review_status = 'published', published_product_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (product_id, now, candidate["id"]),
            )
        conn.commit()
    return {"model": model, "product_id": product_id, "published_count": len(candidates)}


def get_candidate_status(db_path: Optional[Path] = None) -> Dict[str, Any]:
    with connect(db_path) as conn:
        apply_migrations(conn)
        counts = {row["review_status"]: row["count"] for row in conn.execute(
            "SELECT review_status, COUNT(*) count FROM product_spec_candidates GROUP BY review_status"
        ).fetchall()}
        return {
            "candidate_count": sum(counts.values()),
            "pending_count": counts.get("pending", 0),
            "approved_count": counts.get("approved", 0),
            "rejected_count": counts.get("rejected", 0),
            "published_count": counts.get("published", 0),
        }
