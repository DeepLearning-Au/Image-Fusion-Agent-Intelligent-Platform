from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from catalog.repository import get_product, list_products, search_products
from config import settings
from rag.device_query import DeviceQuery, parse_device_query
from rag.enterprise_retrieval import enterprise_search


DISPLAY_SPECS = [
    "infrared_resolution", "visible_resolution", "resolution_width",
    "resolution_height", "stream_frame_rate", "sdk_api_support",
    "network_protocols", "interface_standards", "data_interface", "ip_rating",
    "operating_temperature",
]

RELAXATION_LABELS = {
    "modality": "设备模态",
    "model": "指定型号",
    "resolution": "最低分辨率",
    "frame_rate": "最低帧率",
    "sdk": "SDK/API要求",
    "protocols": "必需协议",
    "ip_rating": "防护等级",
    "fusion": "融合适配要求",
    "price": "预算范围",
}


def _structured_search_kwargs(
    query: DeviceQuery,
    db_path: Optional[Any],
    *,
    omit: Optional[set[str]] = None,
) -> Dict[str, Any]:
    omit = omit or set()
    hard_price = query.price_constraint_hard
    return {
        "modality": None if "modality" in omit else query.modality,
        "min_width": None if "resolution" in omit else query.min_width,
        "min_height": None if "resolution" in omit else query.min_height,
        "min_fps": None if "frame_rate" in omit else query.min_fps,
        "sdk_required": None if "sdk" in omit else query.sdk_required,
        "model": None if "model" in omit else query.model,
        "required_protocols": [] if "protocols" in omit else query.required_protocols,
        "min_ip_rating": None if "ip_rating" in omit else query.min_ip_rating,
        "fusion_required": False if "fusion" in omit else query.fusion_required,
        "min_price": None if "price" in omit or not hard_price else query.min_price,
        "max_price": None if "price" in omit or not hard_price else query.max_price,
        "currency": None if "price" in omit or not hard_price else query.currency,
        "db_path": db_path,
    }


def _active_constraint_groups(query: DeviceQuery) -> List[str]:
    groups = []
    if query.modality:
        groups.append("modality")
    if query.model:
        groups.append("model")
    if query.min_width is not None or query.min_height is not None:
        groups.append("resolution")
    if query.min_fps is not None:
        groups.append("frame_rate")
    if query.sdk_required is not None:
        groups.append("sdk")
    if query.required_protocols:
        groups.append("protocols")
    if query.min_ip_rating is not None:
        groups.append("ip_rating")
    if query.fusion_required:
        groups.append("fusion")
    if query.price_constraint_hard and (query.min_price is not None or query.max_price is not None):
        groups.append("price")
    return groups


def _relaxation_suggestions(query: DeviceQuery, db_path: Optional[Any]) -> List[Dict[str, Any]]:
    """无结果时每次只移除一组硬条件，给出可解释、不会偷偷放宽的建议。"""
    suggestions = []
    for group in _active_constraint_groups(query):
        rows = search_products(**_structured_search_kwargs(query, db_path, omit={group}))
        if not rows:
            continue
        suggestions.append({
            "constraint": group,
            "label": RELAXATION_LABELS[group],
            "match_count": len(rows),
            "models": [row["model"] for row in rows[:5]],
        })
    return sorted(
        suggestions,
        key=lambda item: (-int(item["match_count"]), str(item["constraint"])),
    )


def _value_text(spec: Dict[str, Any]) -> str:
    value = spec.get("value")
    if value is True:
        value = "支持"
    elif value is False:
        value = "不支持"
    elif isinstance(value, list):
        value = "、".join(str(item) for item in value)
    return f"{value}{spec.get('unit') or ''}"


def _product_result(product: Dict[str, Any], query: DeviceQuery) -> Dict[str, Any]:
    specs = {item["spec_key"]: item for item in product.get("specs", [])}
    selected = []
    for key in DISPLAY_SPECS:
        item = specs.get(key)
        if item:
            selected.append({
                "spec_key": key,
                "label": item["label"],
                "value": item["value"],
                "unit": item.get("unit"),
                "source_page": item.get("source_page"),
                "source_document": item.get("document_title"),
            })

    scenario_text = " ".join(
        f"{item.get('scenario', '')} {item.get('note', '')}"
        for item in product.get("scenarios", [])
    )
    scenario_hits = [
        term for term in query.scenarios
        if term in scenario_text or term in product.get("summary", "")
    ]
    compatibility = product.get("fusion_compatibility") or {}
    unknowns = []
    if query.fusion_required:
        for field, label in [
            ("hardware_trigger", "硬件同步"),
            ("timestamp_sync", "时间戳同步"),
            ("calibration_data", "标定数据"),
        ]:
            if compatibility.get(field) is None:
                unknowns.append(label)

    price = product.get("prices", [None])[0] if product.get("prices") else None
    result = {
        "product_id": product["id"],
        "model": product["model"],
        "name": product["name"],
        "manufacturer": product["manufacturer_name"],
        "modality": product["modality"],
        "summary": product.get("summary", ""),
        "specs": selected,
        "scenarios": product.get("scenarios", []),
        "scenario_hits": scenario_hits,
        "price": price,
        "fusion_compatibility": compatibility,
        "needs_vendor_confirmation": unknowns,
        "primary_source": {
            "document": product.get("document_title"),
            "page": product.get("primary_source_page"),
            "path": product.get("source_path"),
            "url": product.get("source_url"),
        },
    }
    result["search_text"] = _build_product_card(result)
    result["structured_preference_score"] = _structured_preference_score(result, query)
    return result


def _build_product_card(product: Dict[str, Any]) -> str:
    spec_text = "；".join(
        f"{item.get('label', '')}:{_value_text(item)}" for item in product.get("specs", [])
    )
    scenario_text = "；".join(
        f"{item.get('scenario', '')}:{item.get('note', '')}"
        for item in product.get("scenarios", [])
    )
    price = product.get("price") or {}
    if price.get("amount_min") is not None:
        maximum = price.get("amount_max")
        maximum = maximum if maximum is not None else price.get("amount_min")
        price_text = f"{price.get('currency') or ''} {price.get('amount_min')}-{maximum}"
    else:
        price_text = "需询价或价格未知"
    return "\n".join([
        f"型号：{product.get('model', '')}",
        f"名称：{product.get('name', '')}",
        f"厂商：{product.get('manufacturer', '')}",
        f"模态：{product.get('modality', '')}",
        f"简介：{product.get('summary', '')}",
        f"规格：{spec_text}",
        f"场景：{scenario_text}",
        f"价格：{price_text}",
        f"融合适配：{product.get('fusion_compatibility') or {}}",
    ])


def _structured_preference_score(product: Dict[str, Any], query: DeviceQuery) -> float:
    """硬条件已由SQL门禁；这里只给偏好、场景和资料完整度打分。"""
    score = float(len(product.get("scenario_hits", [])) * 4)
    compact_text = re.sub(r"\s+", "", product.get("search_text", "").upper())
    for protocol in query.preferred_protocols:
        if re.sub(r"\s+", "", protocol.upper()) in compact_text:
            score += 3
    if query.modality and product.get("modality") == query.modality:
        score += 1
    if not product.get("needs_vendor_confirmation"):
        score += 1

    price = product.get("price") or {}
    amount = price.get("amount_max")
    if amount is None:
        amount = price.get("amount_min")
    if query.max_price is not None and amount is not None:
        score += 3 if float(amount) <= query.max_price else max(0.0, 2 - float(amount) / query.max_price)
    return round(score, 6)


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).replace("\\", "/").lower()


def _source_matches_product(source: Dict[str, Any], product: Dict[str, Any]) -> bool:
    source_text = _normalized(" ".join([
        str(source.get("doc_title") or ""),
        str(source.get("source") or ""),
        str(source.get("text") or source.get("content") or ""),
    ]))
    model = _normalized(product.get("model"))
    if model and model in source_text:
        return True

    primary = product.get("primary_source") or {}
    document = _normalized(primary.get("document"))
    if document and document in source_text:
        return True
    source_path = primary.get("path")
    filename = _normalized(Path(str(source_path)).name) if source_path else ""
    return bool(filename and filename in source_text)


def _attach_product_evidence(
    products: List[Dict[str, Any]], sources: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    for product in products:
        evidence = []
        evidence_score = 0.0
        for rank, source in enumerate(sources, start=1):
            if not _source_matches_product(source, product):
                continue
            evidence_score += 1.0 / (60 + rank)
            evidence.append({
                "doc_title": source.get("doc_title"),
                "page": source.get("page") or source.get("source_page"),
                "source": source.get("source"),
                "score": source.get("final_score", source.get("score", 0)),
                "excerpt": (source.get("matched_child_text") or source.get("text") or "")[:500],
            })
        product["evidence"] = evidence[:5]
        product["evidence_score"] = evidence_score
    return products


def _rrf_product_rankings(
    products: List[Dict[str, Any]], *, rrf_k: int = 60
) -> List[Dict[str, Any]]:
    structured_route = sorted(
        products,
        key=lambda item: (item.get("structured_preference_score", 0), item.get("model", "")),
        reverse=True,
    )
    evidence_route = sorted(
        [item for item in products if item.get("evidence")],
        key=lambda item: (item.get("evidence_score", 0), item.get("model", "")),
        reverse=True,
    )
    routes = [structured_route] + ([evidence_route] if evidence_route else [])
    scores: Dict[int, float] = {int(item["product_id"]): 0.0 for item in products}
    route_ranks: Dict[int, Dict[str, int]] = {int(item["product_id"]): {} for item in products}
    route_names = ["structured_preference", "knowledge_evidence"][:len(routes)]
    for route_name, route in zip(route_names, routes):
        for rank, item in enumerate(route, start=1):
            product_id = int(item["product_id"])
            scores[product_id] += 1.0 / (rrf_k + rank)
            route_ranks[product_id][route_name] = rank

    maximum = len(routes) / float(rrf_k + 1)
    ranked = []
    for item in products:
        value = dict(item)
        product_id = int(value["product_id"])
        value["rrf_score"] = round(scores[product_id] / maximum, 6) if maximum else 0.0
        value["final_score"] = value["rrf_score"]
        value["route_ranks"] = route_ranks[product_id]
        value["text"] = value.get("search_text", "")
        ranked.append(value)
    ranked.sort(key=lambda item: (item["rrf_score"], item["model"]), reverse=True)
    return ranked


def _public_ranked_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for rank, item in enumerate(products, start=1):
        value = dict(item)
        value["ranking"] = {
            "rank": rank,
            "structured_preference_score": value.pop("structured_preference_score", 0),
            "rrf_score": value.pop("rrf_score", 0),
            "bge_score": value.pop("reranker_score", None),
            "final_score": value.pop("final_score", 0),
            "route_ranks": value.pop("route_ranks", {}),
            "evidence_count": len(value.get("evidence", [])),
        }
        for internal_key in ["text", "search_text", "evidence_score", "reranker_rank"]:
            value.pop(internal_key, None)
        result.append(value)
    return result


def hybrid_product_search(
    query: str,
    top_k: int = 5,
    db_path: Optional[Any] = None,
    *,
    rag_result: Optional[Dict[str, Any]] = None,
    product_reranker: Any = None,
    enable_product_reranker: Optional[bool] = None,
) -> Dict[str, Any]:
    known_models = [item["model"] for item in list_products(db_path=db_path)]
    parsed = parse_device_query(query, known_models=known_models)
    hard_price = parsed.price_constraint_hard
    matches = search_products(**_structured_search_kwargs(parsed, db_path))
    products = [
        _product_result(get_product(row["model"], db_path=db_path), parsed)
        for row in matches
    ]

    if rag_result is None:
        rag_result = (
            enterprise_search(query, top_k=max(top_k * 3, 10))
            if db_path is None
            else {"sources": [], "confidence": "not_evaluated", "top_score": 0, "retrieval": {}}
        )
    sources = rag_result.get("sources", []) or []
    products = _attach_product_evidence(products, sources)
    ranked = _rrf_product_rankings(products, rrf_k=int(settings.RAG_RRF_K))

    warnings = list(parsed.warnings) + list(rag_result.get("warnings", []) or [])
    should_rerank = (
        bool(settings.RERANKER_ENABLED and db_path is None)
        if enable_product_reranker is None
        else bool(enable_product_reranker)
    )
    product_reranker_used = False
    if ranked and should_rerank:
        try:
            from rag.bge_reranker import rerank_candidates

            ranked = rerank_candidates(
                query,
                ranked,
                top_k=min(top_k, len(ranked)),
                reranker=product_reranker,
            )
            product_reranker_used = True
        except Exception as exc:
            warnings.append(f"产品BGE重排暂不可用，已保留产品RRF排序：{exc}")
            ranked = ranked[:top_k]
    else:
        ranked = ranked[:top_k]

    relaxation_suggestions = []
    if not products:
        relaxation_suggestions = _relaxation_suggestions(parsed, db_path)
        if hard_price and (parsed.min_price is not None or parsed.max_price is not None):
            warnings.append("没有具有公开价格且同时满足预算与其他硬条件的产品")
        else:
            warnings.append("没有产品同时满足全部硬条件")
        if relaxation_suggestions:
            labels = "、".join(item["label"] for item in relaxation_suggestions[:3])
            warnings.append(f"可尝试放宽以下条件：{labels}；系统未自动修改用户要求")
    if parsed.fusion_required and products:
        warnings.append("融合前仍需确认同步、标定、视场和双路图像访问方式")

    return {
        "query": query,
        "parsed_query": parsed.public_dict(),
        "query_diagnostics": {
            "corrections": parsed.corrections,
            "constraint_confidence": parsed.constraint_confidence,
            "ambiguous_constraints": parsed.ambiguous_constraints,
            "needs_confirmation": parsed.needs_confirmation,
        },
        "structured_match_count": len(products),
        "products": _public_ranked_products(ranked),
        "rag": {
            "confidence": rag_result.get("confidence", "low"),
            "top_score": rag_result.get("top_score", 0),
            "sources": sources[:max(top_k, 5)],
        },
        "retrieval": {
            "hard_filter": "mysql" if db_path is None else "sqlite_test",
            "hard_candidate_count": len(products),
            "knowledge_retrieval": "bm25+milvus",
            "knowledge_fusion": "rrf",
            "knowledge_reranker_used": bool(
                (rag_result.get("retrieval") or {}).get("reranker_used")
            ),
            "product_fusion": "rrf",
            "product_reranker": "BGE" if product_reranker_used else "fallback_rrf",
            "product_reranker_used": product_reranker_used,
            "evidence_linked_product_count": sum(bool(item.get("evidence")) for item in products),
        },
        "warnings": warnings,
        "relaxation_suggestions": relaxation_suggestions,
    }


def format_hybrid_result_for_agent(result: Dict[str, Any]) -> str:
    parsed = result["parsed_query"]
    lines = [
        "结构化条件：" + str({
            key: value for key, value in parsed.items()
            if key != "original_query" and value not in (None, False, [], "lookup")
        })
    ]
    for index, product in enumerate(result["products"], start=1):
        ranking = product.get("ranking", {})
        lines.append(
            f"\n【产品{index}】{product['model']}｜{product['name']}｜"
            f"统一排序分={ranking.get('final_score', 0):.4f}"
        )
        lines.append(f"厂商：{product['manufacturer']}")
        for spec in product["specs"]:
            lines.append(
                f"- {spec['label']}：{_value_text(spec)}；"
                f"来源：{spec.get('source_document')} 第{spec.get('source_page')}页"
            )
        if product["scenario_hits"]:
            lines.append("场景命中：" + "、".join(product["scenario_hits"]))
        if product.get("evidence"):
            lines.append(
                "关联证据：" + "、".join(
                    str(item.get("doc_title") or "未知文档") for item in product["evidence"][:3]
                )
            )
        if product["needs_vendor_confirmation"]:
            lines.append("待厂商确认：" + "、".join(product["needs_vendor_confirmation"]))
    if not result["products"]:
        lines.append("没有产品同时满足全部硬条件。")
        for suggestion in result.get("relaxation_suggestions", []):
            lines.append(
                f"若放宽{suggestion['label']}，可找到 {suggestion['match_count']} 款："
                + "、".join(suggestion.get("models", []))
            )

    diagnostics = result.get("query_diagnostics", {})
    if diagnostics.get("needs_confirmation"):
        lines.append("存在低置信度条件，回答前应先向用户确认。")

    sources = result.get("rag", {}).get("sources", [])
    if sources:
        lines.append("\n补充资料：")
        for source in sources[:5]:
            lines.append(f"- {source.get('doc_title')}：{(source.get('text') or '')[:500]}")
    for warning in result.get("warnings", []):
        lines.append("提示：" + warning)
    return "\n".join(lines)
