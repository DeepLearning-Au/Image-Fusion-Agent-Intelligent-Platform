from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


PROTOCOL_ALIASES: Dict[str, List[str]] = {
    "ONVIF": ["ONVIF"],
    "RTSP": ["RTSP"],
    "MQTT": ["MQTT"],
    "USB3 Vision": ["USB3 VISION", "USB 3 VISION"],
    "USB 3.0": ["USB3.0", "USB 3.0", "USB3", "USB 3"],
    "GigE Vision": ["GIGE VISION"],
    "GigE": ["GIGE", "千兆网"],
    "GenICam": ["GENICAM"],
    "Camera Link": ["CAMERA LINK", "CAMERALINK"],
    "CoaXPress": ["COAXPRESS", "CXP"],
    "MIPI": ["MIPI"],
    "HDMI": ["HDMI"],
    "RJ45": ["RJ45", "RJ-45"],
    "Ethernet": ["ETHERNET", "以太网"],
    "PoE": ["POE"],
    "GB/T 28181": ["GB/T 28181", "GBT 28181", "GB 28181"],
    "HTTP": ["HTTP"],
    "TCP": ["TCP"],
    "SDK": ["SDK"],
    "API": ["API"],
}

SCENARIO_TERMS = [
    "电力巡检", "配电台区", "输电线路", "工厂自动化", "物流读码",
    "瓶检", "医药包装", "室外", "夜间", "全天候", "测温",
]


@dataclass
class DeviceQuery:
    original_query: str
    normalized_query: str = ""
    query_type: str = "lookup"
    modality: Optional[str] = None
    model: Optional[str] = None
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    min_fps: Optional[float] = None
    required_protocols: List[str] = field(default_factory=list)
    sdk_required: Optional[bool] = None
    min_ip_rating: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    currency: Optional[str] = None
    price_constraint_hard: bool = False
    fusion_required: bool = False
    scenarios: List[str] = field(default_factory=list)
    preferred_protocols: List[str] = field(default_factory=list)
    corrections: List[Dict[str, Any]] = field(default_factory=list)
    constraint_confidence: Dict[str, float] = field(default_factory=dict)
    ambiguous_constraints: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    needs_confirmation: bool = False

    def public_dict(self) -> dict:
        return asdict(self)


def normalize_query_text(text: str) -> str:
    """统一全角字符和空白，但保留原句供审计与回显。"""
    value = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", value).strip()


def _compact_term(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _technical_spans(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """提取英文/数字技术词及其子短语，用于有限领域词表的模糊纠错。"""
    spans: List[Tuple[str, Tuple[int, int]]] = []
    seen = set()
    pattern = re.compile(
        r"[A-Za-z][A-Za-z0-9./+\-]*(?:[ \t]+[A-Za-z0-9./+\-]+){0,2}"
    )
    for match in pattern.finditer(text):
        words = match.group(0).split()
        offsets = []
        cursor = match.start()
        for word in words:
            start = text.find(word, cursor, match.end())
            offsets.append((word, start, start + len(word)))
            cursor = start + len(word)
        for size in range(1, min(3, len(offsets)) + 1):
            for index in range(0, len(offsets) - size + 1):
                selected = offsets[index:index + size]
                value = text[selected[0][1]:selected[-1][2]]
                key = (value.upper(), selected[0][1], selected[-1][2])
                if key not in seen:
                    seen.add(key)
                    spans.append((value, (selected[0][1], selected[-1][2])))
    return spans


def _fuzzy_protocol_matches(
    text: str,
    exact_protocols: List[str],
    minimum_score: float = 0.84,
) -> List[Dict[str, Any]]:
    """只在协议白名单内纠错，避免对任意自然语言做开放式猜测。"""
    alias_rows = [
        (protocol, alias, _compact_term(alias))
        for protocol, aliases in PROTOCOL_ALIASES.items()
        for alias in aliases
        if len(_compact_term(alias)) >= 5
    ]
    matches: List[Dict[str, Any]] = []
    used_protocols = set(exact_protocols)
    for raw, position in _technical_spans(text):
        compact = _compact_term(raw)
        if len(compact) < 5:
            continue
        best = None
        for protocol, alias, target in alias_rows:
            if protocol in used_protocols or compact == target:
                continue
            if abs(len(compact) - len(target)) > 2:
                continue
            score = SequenceMatcher(None, compact, target).ratio()
            if best is None or score > best[0]:
                best = (score, protocol, alias)
        if best is None or best[0] < minimum_score:
            continue
        score, protocol, _alias = best
        matches.append({
            "kind": "protocol",
            "original": raw,
            "corrected": protocol,
            "confidence": round(float(score), 4),
            "position": position,
        })
        used_protocols.add(protocol)
    return matches


def _fuzzy_model_suggestion(text: str, known_models: List[str]) -> Optional[Dict[str, Any]]:
    """型号中的一个字符可能改变产品含义，因此只建议、不自动作为硬过滤。"""
    best = None
    for raw, position in _technical_spans(text):
        compact = _compact_term(raw)
        if len(compact) < 5:
            continue
        for model in known_models:
            target = _compact_term(model)
            if compact == target or abs(len(compact) - len(target)) > 1:
                continue
            score = SequenceMatcher(None, compact, target).ratio()
            if best is None or score > best[0]:
                best = (score, raw, position, model)
    if best is None or best[0] < 0.80:
        return None
    score, raw, position, model = best
    return {
        "kind": "model",
        "original": raw,
        "suggested": model,
        "confidence": round(float(score), 4),
        "position": position,
        "requires_confirmation": True,
    }


def _suspicious_resolution(width: int, height: int) -> bool:
    smaller, larger = sorted((width, height))
    return smaller < 64 or larger > 20000 or larger / max(smaller, 1) > 8


def _protocol_position(text: str, aliases: List[str]) -> Optional[Tuple[int, int]]:
    upper = text.upper()
    compact = re.sub(r"\s+", "", upper)
    for alias in aliases:
        target = alias.upper()
        direct = upper.find(target)
        if direct >= 0:
            return direct, direct + len(target)
        compact_target = re.sub(r"\s+", "", target)
        if compact_target and compact_target in compact:
            # 紧凑匹配无法可靠映射原始下标，但仍能判断协议存在。
            return 0, 0
    return None


def _is_soft_preference(text: str, position: Tuple[int, int]) -> bool:
    start, end = position
    context = text[max(0, start - 10): min(len(text), end + 8)]
    return any(word in context for word in ["优先", "最好", "可选", "偏好", "尽量", "希望"])


def _price_number(raw: str, unit: str = "") -> float:
    value = float(raw.replace(",", ""))
    normalized = (unit or "").lower()
    if normalized == "万" or normalized == "w":
        value *= 10000
    elif normalized in {"千", "k"}:
        value *= 1000
    return value


def _parse_price(text: str) -> Tuple[Optional[float], Optional[float], Optional[str], bool]:
    """提取人民币价格区间；明确上限/下限是硬条件，约数和左右是软偏好。"""
    number = r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*([万千wWkK]?)"
    currency = "CNY" if re.search(r"(?:￥|¥|元|人民币|RMB|CNY)", text, re.IGNORECASE) else None
    soft = bool(re.search(r"(?:大约|约|左右|上下|差不多)", text))

    range_match = re.search(
        rf"{number}\s*(?:元|人民币|RMB|CNY)?\s*(?:-|~|～|至|到)\s*{number}\s*(?:元|人民币|RMB|CNY)?",
        text,
        re.IGNORECASE,
    )
    if range_match:
        minimum = _price_number(range_match.group(1), range_match.group(2))
        maximum = _price_number(range_match.group(3), range_match.group(4))
        return min(minimum, maximum), max(minimum, maximum), currency or "CNY", not soft

    upper_patterns = [
        rf"(?:预算(?:不超过|最多|上限(?:为)?|控制在)?|不超过|最高|低于|小于|至多)\s*[￥¥]?\s*{number}",
        rf"[￥¥]?\s*{number}\s*(?:元|人民币|RMB|CNY)?\s*(?:以内|以下|封顶)",
    ]
    for pattern in upper_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return None, _price_number(match.group(1), match.group(2)), currency or "CNY", not soft

    lower_patterns = [
        rf"(?:价格)?(?:不低于|至少|最低)\s*[￥¥]?\s*{number}",
        rf"[￥¥]?\s*{number}\s*(?:元|人民币|RMB|CNY)?\s*(?:以上|起)",
    ]
    for pattern in lower_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _price_number(match.group(1), match.group(2)), None, currency or "CNY", not soft

    budget = re.search(rf"预算\s*[￥¥]?\s*{number}", text, re.IGNORECASE)
    if budget:
        return None, _price_number(budget.group(1), budget.group(2)), currency or "CNY", not soft
    return None, None, currency, False


def parse_device_query(query: str, known_models: Optional[List[str]] = None) -> DeviceQuery:
    original_text = (query or "").strip()
    text = normalize_query_text(original_text)
    upper = text.upper()
    result = DeviceQuery(original_query=original_text, normalized_query=text)

    if any(word in text for word in ["推荐", "选型", "选择", "哪款", "适合什么"]):
        result.query_type = "recommendation"
    elif any(word in text for word in ["对比", "比较", "区别"]):
        result.query_type = "comparison"
    elif any(word in text for word in ["融合", "MambaDFuse", "配准"]):
        result.query_type = "fusion_compatibility"

    for model in sorted(known_models or [], key=len, reverse=True):
        if model.upper() in upper:
            result.model = model
            result.constraint_confidence["model"] = 1.0
            break
    if result.model is None:
        model_suggestion = _fuzzy_model_suggestion(text, known_models or [])
        if model_suggestion:
            result.ambiguous_constraints.append(model_suggestion)
            result.warnings.append(
                f"型号“{model_suggestion['original']}”可能是“{model_suggestion['suggested']}”，需确认后才能作为硬条件"
            )
            result.needs_confirmation = True

    if any(word in text for word in ["双模态", "双光", "红外可见光一体"]):
        result.modality = "dual_ir_visible"
    elif "纯红外" in text or ("红外" in text and "可见光" not in text):
        result.modality = "pure_infrared"
    elif "纯可见光" in text or ("可见光" in text and "红外" not in text) or "工业相机" in text:
        result.modality = "pure_visible"
    if result.modality:
        result.constraint_confidence["modality"] = 1.0

    resolution = re.search(r"(\d{2,5})\s*[×xX*]\s*(\d{2,5})", text)
    if resolution:
        width = int(resolution.group(1))
        height = int(resolution.group(2))
        if _suspicious_resolution(width, height):
            result.ambiguous_constraints.append({
                "kind": "resolution",
                "original": resolution.group(0),
                "confidence": 0.35,
                "requires_confirmation": True,
            })
            result.warnings.append(
                f"分辨率“{resolution.group(0)}”超出常见设备范围，未作为硬条件，请确认是否输入有误"
            )
            result.needs_confirmation = True
        else:
            result.min_width = width
            result.min_height = height
            result.constraint_confidence["resolution"] = 1.0

    fps = re.search(r"(\d+(?:\.\d+)?)\s*(?:FPS|帧|HZ)", upper)
    if fps:
        result.min_fps = float(fps.group(1))
        result.constraint_confidence["min_fps"] = 1.0

    protocols = []
    preferred_protocols = []
    for protocol, aliases in PROTOCOL_ALIASES.items():
        position = _protocol_position(text, aliases)
        if position is None:
            continue
        target = preferred_protocols if _is_soft_preference(text, position) else protocols
        if protocol not in target:
            target.append(protocol)
        result.constraint_confidence[f"protocol:{protocol}"] = 1.0

    exact_protocols = list(dict.fromkeys(protocols + preferred_protocols))
    for correction in _fuzzy_protocol_matches(text, exact_protocols):
        protocol = correction["corrected"]
        is_soft = _is_soft_preference(text, correction["position"])
        confidence = float(correction["confidence"])
        if confidence >= 0.90:
            target = preferred_protocols if is_soft else protocols
            if protocol not in target:
                target.append(protocol)
            correction["applied_as"] = "soft_preference" if is_soft else "hard_filter"
            result.constraint_confidence[f"protocol:{protocol}"] = confidence
            result.warnings.append(
                f"检测到“{correction['original']}”，已按“{protocol}”解析"
            )
        else:
            correction["applied_as"] = "confirmation"
            correction["requires_confirmation"] = True
            result.ambiguous_constraints.append(dict(correction))
            result.warnings.append(
                f"“{correction['original']}”可能是“{protocol}”，置信度不足，未作为硬条件"
            )
            result.needs_confirmation = True
        result.corrections.append(correction)
    # 更具体的工业标准已命中时，不重复追加其通用物理接口。
    if "USB3 Vision" in protocols or "USB3 Vision" in preferred_protocols:
        protocols = [item for item in protocols if item != "USB 3.0"]
        preferred_protocols = [item for item in preferred_protocols if item != "USB 3.0"]
    if "GigE Vision" in protocols or "GigE Vision" in preferred_protocols:
        protocols = [item for item in protocols if item != "GigE"]
        preferred_protocols = [item for item in preferred_protocols if item != "GigE"]
    # SDK/API由专门布尔字段精确筛选，不再重复作为协议LIKE条件。
    sdk_negated = bool(re.search(r"(?:无需|不需要|不要求).{0,5}(?:SDK|API)", text, re.IGNORECASE))
    result.sdk_required = True if any(item in protocols for item in ["SDK", "API"]) and not sdk_negated else None
    result.required_protocols = [item for item in protocols if item not in {"SDK", "API"}]
    result.preferred_protocols = list(preferred_protocols)

    ip_rating = re.search(r"\bIP\s*(\d{2})\b", upper)
    if ip_rating:
        rating_text = ip_rating.group(1)
        if int(rating_text[0]) <= 6:
            result.min_ip_rating = int(rating_text)
            result.constraint_confidence["min_ip_rating"] = 1.0
        else:
            result.ambiguous_constraints.append({
                "kind": "ip_rating",
                "original": ip_rating.group(0),
                "confidence": 0.2,
                "requires_confirmation": True,
            })
            result.warnings.append(
                f"防护等级“{ip_rating.group(0)}”不符合常见IP等级格式，未作为硬条件"
            )
            result.needs_confirmation = True

    (
        result.min_price,
        result.max_price,
        result.currency,
        result.price_constraint_hard,
    ) = _parse_price(text)
    if result.min_price is not None or result.max_price is not None:
        result.constraint_confidence["price"] = 1.0 if result.price_constraint_hard else 0.7

    fusion_negated = bool(re.search(r"(?:无需|不需要|不要求|不做).{0,5}融合", text))
    result.fusion_required = not fusion_negated and any(
        word.upper() in upper for word in ["融合", "MambaDFuse", "双模态", "双光"]
    )
    result.scenarios = [term for term in SCENARIO_TERMS if term in text]
    return result
