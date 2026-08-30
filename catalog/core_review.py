from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from catalog.candidates import list_candidates, publish_approved_candidates, review_candidate
from catalog.repository import apply_product_profile, get_product


REVIEWED_BY = "official_pdf_review_v1"

CORRECTIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "IPT640M": {
        "infrared_resolution_width": {"spec_key": "resolution_width"},
        "infrared_resolution_height": {"spec_key": "resolution_height"},
        "thermal_sensitivity": {"value_type": "number", "value": 50, "unit": "mK@30°C", "comparator": "lte"},
        "infrared_lens": {"spec_key": "lens_focal_lengths", "value_type": "json", "value": [5.9, 15, 25], "unit": "mm"},
        "stream_profile": {"spec_key": "stream_frame_rate", "value_type": "number", "value": 25, "unit": "Hz"},
        "network_protocols": {"value": ["TCP", "UDP", "ICMP", "IGMP", "DHCP", "RTSP", "ONVIF", "GB/T 28181"]},
        "languages": {"value": ["中文", "英文"]},
        "certifications": {"value": ["CE", "EMC", "RoHS"]},
        "power_consumption": {"spec_key": "max_power", "value_type": "number", "value": 2.5, "unit": "W", "comparator": "lte"},
        "weight": {"spec_key": "max_weight", "value_type": "number", "value": 215, "unit": "g", "comparator": "lte"},
    },
    "MV-CA050-12UC": {
        "pixel_pitch": {"value_type": "number", "value": 3.45, "unit": "μm"},
        "dynamic_range": {"value_type": "number", "value": 72, "unit": "dB"},
        "signal_to_noise_ratio": {"value_type": "number", "value": 40.2, "unit": "dB"},
        "power_consumption": {"value_type": "number", "value": 3.3, "unit": "W"},
        "weight": {"value_type": "number", "value": 80, "unit": "g"},
        "certifications": {"value": ["CE", "RoHS", "KC"]},
    },
    "MC2003FT-I-W": {
        "sensor_type": {"value": "1/1.8\" CMOS"},
        "thermal_sensitivity": {"value_type": "number", "value": 50, "unit": "mK@30°C"},
        "network_protocols": {"value": ["IPv4/IPv6", "HTTP", "SMTP", "RTSP", "TCP", "DHCP", "ONVIF", "GB/T 28181", "MQTT", "GB/T 35697-2017", "Q/CSG1205031-2020"]},
        "languages": {"value": ["中文", "英文"]},
        "illuminators": {"value": ["红外补光", "白光补光"]},
        "power_consumption": {"value_type": "number", "value": 12, "unit": "W"},
        "weight": {"value_type": "number", "value": 3.6, "unit": "kg"},
        "certifications": {"value": ["中国电科院型式试验"]},
    },
}

REJECT_KEYS = {
    "IPT640M": set(),
    "MV-CA050-12UC": set(),
    # 原表写“分4级”但只提取出两个名称，信息不完整，暂不发布。
    "MC2003FT-I-W": {"user_management"},
}

PROFILES: Dict[str, Dict[str, Any]] = {
    "MV-CA050-12UC": {
        "model": "MV-CA050-12UC",
        "name": "MV-CA050-12UC 500万像素USB3.0彩色工业面阵相机",
        "modality": "pure_visible",
        "summary": "采用Sony IMX264全局快门传感器的彩色工业相机，支持2448×2048分辨率、60 fps、USB3 Vision与GenICam。",
        "source_url": "https://www.hikrobotics.com/cn2/source/vision/document/2023/6/9/MV-CA050-12UMUC_20230510.pdf",
        "price": {"price_type": "inquiry", "source_page": 2, "note": "官方资料未提供公开价格，需向厂商或经销商询价。"},
        "media": [{"media_type": "product_image", "uri": "/knowledge-assets/documents/ecffeaa1235d1defdd7d/page_001_image_04_7a995615.jpeg", "caption": "MV-CA050-12UC官方资料中的相机外观图。", "source_page": 1}],
        "scenarios": [
            {"scenario": "电子半导体与工厂自动化", "source_page": 1},
            {"scenario": "物流读码", "source_page": 1},
            {"scenario": "饮料瓶检与医药包装", "source_page": 1},
        ],
        "fusion_compatibility": {
            "mambadfuse_role": "visible_source",
            "infrared_stream_access": False,
            "visible_stream_access": True,
            "independent_modal_streams": False,
            "hardware_trigger": True,
            "timestamp_sync": None,
            "sdk_access": None,
            "calibration_data": None,
            "registration_required": True,
            "suitability_status": "conditional",
            "notes": "可作为MambaDFuse可见光输入源；需另配红外相机，并确认同步、标定和配准方案。",
            "source_page": 1,
        },
    },
    "MC2003FT-I-W": {
        "model": "MC2003FT-I-W",
        "name": "MC2003FT-I-W 红外可见光双模态枪机",
        "modality": "dual_ir_visible",
        "summary": "面向室外配电台区和输电线路的双模态枪机，提供256×192红外成像、1920×1080可见光成像、测温和双光融合能力。",
        "source_url": "https://www.guideir.cn/Cn/Skippower/downloadFile?id=530&mid=45",
        "price": {"price_type": "inquiry", "source_page": 2, "note": "官方资料未提供公开价格，需向厂商或经销商询价。"},
        "media": [{"media_type": "product_image", "uri": "/knowledge-assets/documents/314dfeb01431b432a76a/page_001_image_03_c9cd8e1f.jpeg", "caption": "MC-FT系列官方资料中的双模态枪机外观图。", "source_page": 1}],
        "scenarios": [
            {"scenario": "室外配电台区监控与测温", "source_page": 1},
            {"scenario": "输电线路监控", "source_page": 1},
            {"scenario": "恶劣天气全天候监控", "source_page": 1},
        ],
        "fusion_compatibility": {
            "mambadfuse_role": "dual_source_candidate",
            "infrared_stream_access": True,
            "visible_stream_access": True,
            "independent_modal_streams": True,
            "hardware_trigger": None,
            "timestamp_sync": None,
            "sdk_access": True,
            "calibration_data": None,
            "registration_required": True,
            "suitability_status": "conditional",
            "notes": "资料显示红外与可见光码流及双光融合能力；接入MambaDFuse前仍需确认同步时间戳、标定参数和原始双路图像访问方式。",
            "source_page": 2,
        },
    },
}


def review_and_publish_core_products(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(db_path) if db_path is not None else None
    results: Dict[str, Any] = {}
    for model in CORRECTIONS:
        pending = list_candidates(status="pending", model=model, db_path=path)
        approved = 0
        rejected = 0
        for candidate in pending:
            key = candidate["spec_key"]
            if key in REJECT_KEYS.get(model, set()):
                review_candidate(
                    candidate["id"], "rejected", reviewed_by=REVIEWED_BY,
                    review_note="PDF表格信息不完整，暂不进入正式产品库", db_path=path,
                )
                rejected += 1
                continue
            updates = dict(CORRECTIONS.get(model, {}).get(key, {}))
            review_candidate(
                candidate["id"], "approved", updates=updates,
                reviewed_by=REVIEWED_BY, review_note="已对照官方PDF表格核验",
                db_path=path,
            )
            approved += 1

        approved_now = list_candidates(status="approved", model=model, db_path=path)
        if approved_now:
            published = publish_approved_candidates(model, db_path=path)
        else:
            product = get_product(model, db_path=path)
            published = {
                "model": model,
                "product_id": product.get("id"),
                "published_count": 0,
            }
        if model in PROFILES:
            apply_product_profile(PROFILES[model], db_path=path)
        results[model] = {
            "approved": approved,
            "rejected": rejected,
            **published,
        }
    return results
