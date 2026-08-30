# rag/eval_service.py
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        return []

    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    return rows


def save_default_eval_set(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    default_items = build_default_eval_items()

    with path.open("w", encoding="utf-8") as f:
        for item in default_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_default_eval_items() -> List[Dict[str, Any]]:
    """返回覆盖产品、融合、故障与维护场景的中文检索评测集。"""
    ir_doc = "GuideIR_IPT640M纯红外测温机芯.pdf"
    visible_doc = "HIKROBOT_MV-CA050-12UC纯可见光工业相机.pdf"
    dual_doc = "GuideIR_MC-FT红外可见光双模态枪机.pdf"
    return [
        {"id": "ir_resolution_fps", "intent": "纯红外设备选型", "question": "哪款纯红外设备支持640×512分辨率和25Hz码流？", "expected_docs": [ir_doc], "keywords": ["IPT640M", "640×512", "25Hz"]},
        {"id": "ir_temperature_range", "intent": "纯红外设备参数", "question": "IPT640M的测温范围和测温精度是多少？", "expected_docs": [ir_doc], "keywords": ["-20°C", "550°C", "±2"]},
        {"id": "ir_sdk_protocol", "intent": "纯红外设备集成", "question": "哪款红外测温机芯支持RTSP、ONVIF以及SDK API集成？", "expected_docs": [ir_doc], "keywords": ["RTSP", "ONVIF", "SDK"]},
        {"id": "ir_lens", "intent": "纯红外设备参数", "question": "IPT640M提供哪些焦距的热成像镜头？", "expected_docs": [ir_doc], "keywords": ["5.9mm", "15mm", "25mm"]},
        {"id": "visible_sensor", "intent": "纯可见光设备选型", "question": "哪款可见光工业相机采用Sony IMX264全局快门并具有2448×2048分辨率？", "expected_docs": [visible_doc], "keywords": ["Sony IMX264", "全局快门", "2448×2048"]},
        {"id": "visible_color_fps", "intent": "纯可见光设备参数", "question": "MV-CA050-12UC彩色型号在Bayer RG 8格式下最大帧率是多少？", "expected_docs": [visible_doc], "keywords": ["60 fps", "Bayer RG 8", "MV-CA050-12UC"]},
        {"id": "visible_trigger", "intent": "纯可见光设备集成", "question": "哪款可见光工业相机同时支持硬触发、软触发和USB3 Vision？", "expected_docs": [visible_doc], "keywords": ["硬触发", "软触发", "USB3 Vision"]},
        {"id": "visible_pixel_formats", "intent": "纯可见光设备参数", "question": "MV-CA050-12UC可以输出哪些彩色像素格式？", "expected_docs": [visible_doc], "keywords": ["Bayer", "YUV", "RGB8"]},
        {"id": "dual_resolution", "intent": "双模态设备选型", "question": "哪款设备同时具备红外和可见光，两个模态的分辨率分别是多少？", "expected_docs": [dual_doc], "keywords": ["256×192", "1920×1080", "MC2003FT-I-W"]},
        {"id": "dual_environment", "intent": "双模态设备场景", "question": "MC-FT双模态枪机的防护等级和工作温度范围是什么？", "expected_docs": [dual_doc], "keywords": ["IP66", "-30°C", "+60°C"]},
        {"id": "dual_integration", "intent": "双模态设备集成", "question": "哪款红外可见光双模态设备支持RTSP、ONVIF、MQTT和开放式SDK API？", "expected_docs": [dual_doc], "keywords": ["RTSP", "MQTT", "SDK/API"]},
        {"id": "dual_fusion", "intent": "融合适配判断", "question": "哪款产品具备红外、可见光码流并支持双光融合？", "expected_docs": [dual_doc], "keywords": ["热成像", "可见光", "双光融合"]},
        {"id": "sar_noise_fault", "intent": "SAR故障诊断", "question": "SAR图像颗粒感强、噪声大时应该检查什么？", "expected_docs": ["SAR设备常见故障案例.md"], "keywords": ["相干斑", "噪声", "滤波"]},
        {"id": "sar_distortion", "intent": "SAR故障诊断", "question": "SAR图像出现几何拉伸或目标位置偏移如何排查？", "expected_docs": ["SAR设备常见故障案例.md"], "keywords": ["几何畸变", "姿态", "配准"]},
        {"id": "sar_rain_fault", "intent": "SAR故障诊断", "question": "雨后SAR设备信号异常应重点检查哪些位置？", "expected_docs": ["SAR设备常见故障案例.md"], "keywords": ["天线罩", "接口", "进水"]},
        {"id": "sar_all_weather", "intent": "SAR能力介绍", "question": "为什么SAR适合全天候和雨雾环境成像？", "expected_docs": ["SAR雷达成像设备产品说明书.md"], "keywords": ["全天候", "雨雾", "微波"]},
        {"id": "sar_limits", "intent": "SAR能力介绍", "question": "SAR成像常见局限有哪些？", "expected_docs": ["SAR雷达成像设备产品说明书.md"], "keywords": ["相干斑", "几何畸变", "配准"]},
        {"id": "fusion_inputs", "intent": "融合服务使用", "question": "运行光学与SAR融合服务需要准备哪些输入？", "expected_docs": ["WEMFusion光学SAR融合服务说明.md"], "keywords": ["光学图像", "SAR图像", "配准"]},
        {"id": "fusion_qabf_low", "intent": "融合质量诊断", "question": "融合结果Qabf偏低通常说明什么问题？", "expected_docs": ["WEMFusion光学SAR融合服务说明.md", "融合指标说明.md"], "keywords": ["边缘", "配准", "噪声"]},
        {"id": "fusion_ghosting", "intent": "融合质量诊断", "question": "融合图像出现双边缘和重影应如何排查？", "expected_docs": ["WEMFusion光学SAR融合服务说明.md"], "keywords": ["重影", "配准"]},
        {"id": "optical_blur", "intent": "光学故障诊断", "question": "可见光图像整体模糊应该检查什么？", "expected_docs": ["光学设备常见故障案例.md"], "keywords": ["对焦", "镜头", "模糊"]},
        {"id": "optical_color", "intent": "光学故障诊断", "question": "可见光画面明显偏色如何处理？", "expected_docs": ["光学设备常见故障案例.md"], "keywords": ["白平衡", "偏色"]},
        {"id": "optical_lowlight", "intent": "光学故障诊断", "question": "低照度下画面过暗且噪声大怎么排查？", "expected_docs": ["光学设备常见故障案例.md"], "keywords": ["低照度", "曝光", "噪声"]},
        {"id": "selection_night", "intent": "设备选型", "question": "夜间监控应选光学、SAR还是融合方案？", "expected_docs": ["光学设备与SAR设备选购指南.md"], "keywords": ["夜间", "SAR", "融合"]},
        {"id": "selection_coast", "intent": "设备选型", "question": "海边高湿盐雾环境选设备要关注哪些条件？", "expected_docs": ["光学设备与SAR设备选购指南.md"], "keywords": ["海边", "盐雾", "防护"]},
        {"id": "maintenance_daily", "intent": "设备维护", "question": "成像设备日常巡检应该检查哪些内容？", "expected_docs": ["成像设备维护保养手册.md"], "keywords": ["镜头", "接口", "测试图像"]},
        {"id": "maintenance_storage", "intent": "设备维护", "question": "成像设备长期存放需要什么环境？", "expected_docs": ["成像设备维护保养手册.md"], "keywords": ["干燥", "通风", "高温"]},
        {"id": "weather_rain", "intent": "环境维护", "question": "雨雪天气使用成像设备要做哪些防护？", "expected_docs": ["天气条件下设备维护规范.md"], "keywords": ["防水", "镜头", "天线罩"]},
        {"id": "weather_salt", "intent": "环境维护", "question": "盐雾环境如何防止成像设备腐蚀？", "expected_docs": ["天气条件下设备维护规范.md"], "keywords": ["盐雾", "腐蚀", "防腐"]},
        {"id": "metric_mi", "intent": "融合指标解释", "question": "融合评价中的互信息MI代表什么，偏低如何改进？", "expected_docs": ["融合指标说明.md"], "keywords": ["互信息", "配准", "跨模态"]},
    ]


def evaluate_retrieval(
    eval_items: List[Dict[str, Any]],
    search_fn: Callable[[str], Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    details = []
    doc_hit_count = 0
    top1_hit_count = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    keyword_coverage_sum = 0.0
    intent_stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"total": 0, "doc_hits": 0, "keyword_coverage_sum": 0.0}
    )

    for item in eval_items:
        question = item.get("question", "")
        expected_docs = set(item.get("expected_docs", []))
        expected_keywords = item.get("keywords", [])

        search_result = search_fn(question)
        sources = search_result.get("sources", [])[:top_k]

        hit_docs = [s.get("doc_title", "") for s in sources]
        all_text = "\n".join([
            (s.get("text", "") or "") + " " + (s.get("doc_title", "") or "")
            for s in sources
        ])

        doc_rank = None
        for i, doc in enumerate(hit_docs, start=1):
            if doc in expected_docs:
                doc_rank = i
                break

        counted_docs = set()
        relevance = []
        for doc in hit_docs:
            relevant = doc in expected_docs and doc not in counted_docs
            relevance.append(1 if relevant else 0)
            if relevant:
                counted_docs.add(doc)
        dcg = sum(
            rel / math.log2(rank + 1)
            for rank, rel in enumerate(relevance, start=1)
        )
        ideal_count = min(len(expected_docs), top_k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg = dcg / idcg if idcg else 0.0

        keyword_hits = []
        for kw in expected_keywords:
            if kw.lower() in all_text.lower():
                keyword_hits.append(kw)

        is_doc_hit = doc_rank is not None
        keyword_coverage = len(keyword_hits) / len(expected_keywords) if expected_keywords else 1.0

        if is_doc_hit:
            doc_hit_count += 1
            mrr_sum += 1.0 / doc_rank
            if doc_rank == 1:
                top1_hit_count += 1
        keyword_coverage_sum += keyword_coverage
        ndcg_sum += ndcg

        intent = item.get("intent", "未分类")
        intent_stats[intent]["total"] += 1
        intent_stats[intent]["doc_hits"] += int(is_doc_hit)
        intent_stats[intent]["keyword_coverage_sum"] += keyword_coverage

        details.append({
            "编号": item.get("id", ""),
            "意图": intent,
            "问题": question,
            "期望文档": "；".join(expected_docs),
            "命中文档": "；".join(hit_docs),
            "期望关键词": "；".join(expected_keywords),
            "命中关键词": "；".join(keyword_hits),
            "是否命中文档": bool(is_doc_hit),
            "文档排名": doc_rank if doc_rank is not None else "",
            "关键词覆盖率": round(keyword_coverage, 4),
            "nDCG": round(ndcg, 4),
            "置信度": search_result.get("confidence", ""),
            "最高得分": round(float(search_result.get("top_score", 0)), 4),
        })

    total = len(eval_items)

    by_intent = {}
    for intent, stats in intent_stats.items():
        count = int(stats["total"])
        by_intent[intent] = {
            "total": count,
            "doc_hit_rate_at_k": stats["doc_hits"] / count if count else 0.0,
            "keyword_coverage": stats["keyword_coverage_sum"] / count if count else 0.0,
        }

    doc_hit_rate = doc_hit_count / total if total else 0.0
    return {
        "total": total,
        # 保留旧字段名，兼容已有调用；现在严格按期望文档是否命中计算。
        "recall_at_k": doc_hit_rate,
        "doc_hit_rate_at_k": doc_hit_rate,
        "top1_accuracy": top1_hit_count / total if total else 0.0,
        "mrr": mrr_sum / total if total else 0.0,
        "ndcg_at_k": ndcg_sum / total if total else 0.0,
        "keyword_coverage": keyword_coverage_sum / total if total else 0.0,
        "by_intent": by_intent,
        "details": details,
    }
