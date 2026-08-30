import json
import urllib.parse
import urllib.request
from typing import Dict, Optional

from langchain.tools import tool

from config import settings
from rag.rag_service import rag_retrieve
from rag.hybrid_product_search import format_hybrid_result_for_agent, hybrid_product_search
from reports.report_generator import generate_report

import re


COMMON_CITY_COORDS = {
    "上海": (31.2304, 121.4737, "上海", "中国"),
    "烟台": (37.4638, 121.4479, "烟台", "中国"),
    "北京": (39.9042, 116.4074, "北京", "中国"),
    "广州": (23.1291, 113.2644, "广州", "中国"),
    "深圳": (22.5431, 114.0579, "深圳", "中国"),
    "杭州": (30.2741, 120.1551, "杭州", "中国"),
    "南京": (32.0603, 118.7969, "南京", "中国"),
    "青岛": (36.0671, 120.3826, "青岛", "中国"),
    "济南": (36.6512, 117.1201, "济南", "中国"),
    "天津": (39.3434, 117.3616, "天津", "中国"),
    "苏州": (31.2989, 120.5853, "苏州", "中国"),
    "宁波": (29.8683, 121.5440, "宁波", "中国"),
    "成都": (30.5728, 104.0668, "成都", "中国"),
    "武汉": (30.5928, 114.3055, "武汉", "中国"),
    "西安": (34.3416, 108.9398, "西安", "中国"),
}


def extract_city(text: str, default_city: str = "烟台") -> str:
    """
    从完整用户问题中提取城市。
    例如：
    上海 明天天气及设备进水维护 -> 上海
    上海明天的天气怎么样 -> 上海
    我现在的城市是上海，设备进水怎么维护 -> 上海
    """
    text = text or ""
    text = str(text)

    for city in COMMON_CITY_COORDS:
        if city in text:
            return city

    match = re.search(r"城市是([\u4e00-\u9fa5]{2,6})", text)
    if match:
        return match.group(1)

    match = re.search(r"([\u4e00-\u9fa5]{2,6})(?:今天|明天|后天)?的?天气", text)
    if match:
        candidate = match.group(1)
        candidate = candidate.replace("我现在的", "").replace("当前", "")
        return candidate

    return default_city


def extract_day_offset(text: str) -> int:
    """
    判断查询今天、明天还是后天。
    """
    text = text or ""
    text = str(text)

    if "后天" in text:
        return 2

    if "明天" in text:
        return 1

    return 0


def query_weather(city_or_requirement: str) -> str:
    """
    查询天气。
    支持完整问题输入，例如：
    上海 明天天气及设备进水维护
    上海明天的天气怎么样，设备进水怎么维护
    """
    try:
        raw_text = city_or_requirement or settings.AGENT_USER_CITY
        raw_text = str(raw_text)

        city = extract_city(raw_text, default_city=settings.AGENT_USER_CITY)
        day_offset = extract_day_offset(raw_text)

        if city in COMMON_CITY_COORDS:
            lat, lon, name, country = COMMON_CITY_COORDS[city]
        else:
            geo_url = (
                "https://geocoding-api.open-meteo.com/v1/search?"
                f"name={urllib.parse.quote(city)}&count=1&language=zh&format=json"
            )

            with urllib.request.urlopen(geo_url, timeout=10) as resp:
                geo_data = json.loads(resp.read().decode("utf-8"))

            results = geo_data.get("results") or []

            if not results:
                return f"未查询到城市：{city} 的天气数据。"

            loc = results[0]
            lat = loc["latitude"]
            lon = loc["longitude"]
            name = loc.get("name", city)
            country = loc.get("country", "")

        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,wind_speed_10m_max"
            "&forecast_days=3"
            "&timezone=auto"
        )

        with urllib.request.urlopen(weather_url, timeout=10) as resp:
            weather_data = json.loads(resp.read().decode("utf-8"))

        current = weather_data.get("current", {})
        daily = weather_data.get("daily", {})

        current_temp = current.get("temperature_2m", "未知")
        current_humidity = current.get("relative_humidity_2m", "未知")
        current_precipitation = current.get("precipitation", "未知")
        current_rain = current.get("rain", "未知")
        current_wind = current.get("wind_speed_10m", "未知")

        dates = daily.get("time", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        precipitation_sum = daily.get("precipitation_sum", [])
        rain_sum = daily.get("rain_sum", [])
        wind_max = daily.get("wind_speed_10m_max", [])

        if day_offset < len(dates):
            target_date = dates[day_offset]
            target_tmax = tmax[day_offset]
            target_tmin = tmin[day_offset]
            target_precipitation = precipitation_sum[day_offset]
            target_rain = rain_sum[day_offset]
            target_wind = wind_max[day_offset]
        else:
            target_date = "未知"
            target_tmax = "未知"
            target_tmin = "未知"
            target_precipitation = "未知"
            target_rain = "未知"
            target_wind = "未知"

        day_name = "今天" if day_offset == 0 else "明天" if day_offset == 1 else "后天"

        return (
            f"城市：{name} {country}\n"
            f"查询日期：{day_name}，{target_date}\n\n"
            f"当前天气：\n"
            f"- 当前温度：{current_temp} ℃\n"
            f"- 当前相对湿度：{current_humidity}%\n"
            f"- 当前降水量：{current_precipitation} mm\n"
            f"- 当前降雨量：{current_rain} mm\n"
            f"- 当前风速：{current_wind} km/h\n\n"
            f"{day_name}预报：\n"
            f"- 最高温度：{target_tmax} ℃\n"
            f"- 最低温度：{target_tmin} ℃\n"
            f"- 总降水量：{target_precipitation} mm\n"
            f"- 总降雨量：{target_rain} mm\n"
            f"- 最大风速：{target_wind} km/h"
        )

    except Exception as e:
        return f"天气查询失败：{e}"

def build_wemfusion_tools(
    last_result: Optional[Dict] = None,
    last_report: Optional[str] = None,
    optical_path: Optional[str] = None,
    sar_path: Optional[str] = None,
    use_tile: bool = False,
    tile: int = 256,
    overlap: int = 32,
    user_city: str = "烟台",
    tool_state: Optional[Dict] = None,
):
    """
    构建 Qwen Agent 可调用工具。

    重点：
    wemfusion_fusion_tool 会真正调用你的本地 WEMFusion / MambaDFuse 模型。
    也就是说，后续是：
    用户自然语言 -> Qwen 判断 -> 调用 wemfusion_fusion_tool -> run_wemfusion()
    """
    if tool_state is None:
        tool_state = {}

    def get_current_result():
        return tool_state.get("last_result") or last_result

    def get_current_report():
        return tool_state.get("last_report") or last_report

    @tool
    def wemfusion_fusion_tool(user_requirement: str) -> str:
        """
        当用户要求融合红外图像和可见光图像、生成融合图、运行MambaDFuse模型、
        分析上传图像、生成融合报告或售后检测报告时，必须调用该工具。
        该工具会调用本地 WEMFusion/MambaDFuse 模型执行图像融合。
        """
        if not optical_path or not sar_path:
            return (
                "无法执行图像融合：当前没有同时检测到可见光图像和红外图像。"
                "请先上传同一场景的可见光图像与红外图像。"
            )

        # 融合模型按需加载，产品检索不应依赖PyTorch/Mamba运行环境。
        from fusion.wemfusion_runner import run_wemfusion

        result = run_wemfusion(
            optical_path=optical_path,
            sar_path=sar_path,
            output_dir=str(settings.OUTPUT_DIR),
            use_tile=use_tile,
            tile=tile,
            overlap=overlap,
        )

        report = generate_report(result, user_note=user_requirement)

        tool_state["last_result"] = result
        tool_state["last_report"] = report

        metrics = result["metrics"]

        return f"""
MambaDFuse 红外—可见光融合模型已成功执行。

输入：
- 可见光图像：{result["optical_path"]}
- 红外图像：{result["sar_path"]}

输出文件：
- 灰度融合图：{result["fused_gray_path"]}
- 彩色融合图：{result["fused_color_path"]}
- 可见光边缘图：{result["edge_optical_path"]}
- 红外边缘图：{result["edge_sar_path"]}
- 融合边缘图：{result["edge_fused_path"]}
- 红外—可见光差异图：{result["diff_path"]}

关键指标：
- EN 信息熵：{metrics.get("EN 信息熵", 0):.4f}
- SD 标准差：{metrics.get("SD 标准差", 0):.4f}
- SF 空间频率：{metrics.get("SF 空间频率", 0):.4f}
- AG 平均梯度：{metrics.get("AG 平均梯度", 0):.4f}
- MI 互信息：{metrics.get("MI 互信息", 0):.4f}
- CC 相关系数：{metrics.get("CC 相关系数", 0):.4f}
- PSNR 峰值信噪比：{metrics.get("PSNR 峰值信噪比", 0):.4f}
- SSIM 结构相似性：{metrics.get("SSIM 结构相似性", 0):.4f}
- Qabf 边缘保持：{metrics.get("Qabf 边缘保持", 0):.4f}
- 综合质量评分：{metrics.get("综合质量评分", 0):.2f}/100

请基于这些工具结果，面向客户总结本次融合质量、可能问题和后续建议。
"""

    @tool
    def rag_search_tool(query: str) -> str:
        """
        当用户询问红外、可见光或双模态产品说明、选购建议、维护、融合指标、
        边缘模糊、配准错位或MambaDFuse模型原理等问题时调用。
        """
        return rag_retrieve(query, top_k=5)

    @tool
    def product_advice_tool(user_need: str) -> str:
        """
        当用户询问红外、可见光和双模态设备介绍、型号选择、购买建议、适用场景时调用。
        """
        context = format_hybrid_result_for_agent(
            hybrid_product_search(user_need, top_k=5)
        )

        return f"""
用户需求：
{user_need}

结构化产品筛选与知识库补充结果：
{context}

请基于资料，从以下方面回答：
1. 应选择纯红外设备、纯可见光设备、双模态设备，还是红外与可见光设备组合；
2. 推荐原因；
3. 适用场景；
4. 需要注意的局限；
5. 是否适合取得两路独立图像并使用 MambaDFuse 融合。
"""

    @tool
    def fault_diagnosis_tool(fault_description: str) -> str:
        """
        当用户描述红外设备、可见光设备或融合图像异常，出现边缘模糊、错位、
        噪声过大、雨天后画面不稳定或需要报修时调用。
        """
        context = rag_retrieve(fault_description, top_k=5)
        result = get_current_result()

        metrics_text = "当前没有融合指标。"

        if result is not None:
            metrics = result.get("metrics", {})
            rows = []

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    rows.append(f"- {k}: {v:.4f}")

            metrics_text = "\n".join(rows)

        return f"""
用户故障描述：
{fault_description}

当前融合指标：
{metrics_text}

知识库故障资料：
{context}

请按以下结构进行售后诊断：
1. 可能原因；
2. 排查步骤；
3. 是否建议报修；
4. 临时处理方案；
5. 后续维护建议。
"""

    @tool
    def weather_maintenance_tool(city_or_requirement: str) -> str:
        """
        当用户询问某城市天气条件下如何维护红外设备或可见光设备、
        雨天/高湿/低温/海边/大风/进水/受潮环境使用注意事项时调用。
        """
        weather = query_weather(city_or_requirement)

        context = rag_retrieve(
            "天气 条件 红外设备 可见光设备 双模态设备 维护 保养 雨天 高湿 进水 受潮 海边 盐雾 低温",
            top_k=5,
        )

        return f"""
    天气查询结果：
    {weather}

    维护知识库资料：
    {context}

    请结合天气条件和设备进水/雨后维护要求，从以下方面给出建议：
    1. 红外成像设备维护；
    2. 可见光成像设备维护；
    3. 设备进水后的紧急处理步骤；
    4. 接口、防水、防潮和密封检查；
    5. 是否适合继续外场使用；
    6. 使用后存放建议。
    """

    @tool
    def current_fusion_metrics_tool(query: str) -> str:
        """
        当用户询问当前融合结果、融合质量、EN、SD、SF、AG、MI、SSIM、Qabf、
        综合评分或当前融合图是否好时调用。
        """
        result = get_current_result()

        if result is None:
            return "当前还没有融合结果。请先调用 wemfusion_fusion_tool 完成一次红外—可见光图像融合。"

        metrics = result.get("metrics", {})

        lines = ["当前融合结果指标如下："]

        for key, value in metrics.items():
            try:
                lines.append(f"- {key}: {float(value):.4f}")
            except Exception:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    @tool
    def fusion_improvement_advice_tool(query: str) -> str:
        """
        当用户询问为什么融合效果不好、Qabf为什么低、边缘为什么模糊、
        如何优化模型、如何提高红外—可见光融合质量时调用。
        """
        result = get_current_result()

        if result is None:
            return "当前没有融合结果，无法基于指标给出针对性建议。"

        metrics = result.get("metrics", {})

        score = metrics.get("综合质量评分", 0)
        qabf = metrics.get("Qabf 边缘保持", 0)
        mi = metrics.get("MI 互信息", 0)
        sf = metrics.get("SF 空间频率", 0)
        ag = metrics.get("AG 平均梯度", 0)

        suggestions = [f"当前综合质量评分为 {score:.2f}/100。"]

        if qabf < 0.6:
            suggestions.append(
                "Qabf 边缘保持指标偏低，说明融合图像对源图像边缘结构的继承能力不足。"
                "建议检查红外—可见光配准误差，并在训练中加入边缘保持损失或梯度损失。"
            )

        if mi < 3.0:
            suggestions.append(
                "MI 互信息偏低，说明融合图像对可见光纹理和红外目标信息保留不足。"
                "建议增强跨模态互补特征提取，并优化DAG-MF差异感知门控。"
            )

        if sf < 20:
            suggestions.append(
                "SF 空间频率偏低，说明纹理细节表达不足。"
                "建议加强高频细节分支，提高可见光真实边缘和方向纹理注入能力。"
            )

        if ag < 10:
            suggestions.append(
                "AG 平均梯度偏低，说明边缘清晰度仍可提升。"
                "建议加入Sobel梯度约束或结构一致性损失。"
            )

        suggestions.append(
            "若融合图出现局部结构漂移，应优先检查两类设备采集图像的配准精度；"
            "若红外噪声明显，建议先排查传感器噪声、非均匀性校正和采集质量。"
        )

        return "\n".join([f"- {s}" for s in suggestions])

    @tool
    def current_report_tool(query: str) -> str:
        """
        当用户要求生成报告、总结本次融合报告、写售后检测结论、写实验结论、
        写项目说明或简历描述时调用。
        """
        report = get_current_report()

        if not report:
            return "当前还没有生成报告。请先调用 wemfusion_fusion_tool 完成一次融合分析。"

        return report[:3500]

    @tool
    def wemfusion_model_info_tool(query: str) -> str:
        """
        当用户询问WEMFusion模型、模型结构、创新点、项目亮点、
        或如何体现大模型Agent调用本地深度学习模型时调用。
        """
        return """
当前系统调用的是本地 MambaDFuse 红外—可见光图像融合模型。

模型核心亮点：
1. DWT/IDWT 小波分解重构：显式分离低频结构和高频方向细节；
2. 融合网络同时保留可见光纹理与红外显著目标信息；
3. Mamba 模块用于建模红外与可见光跨模态的长程依赖；
4. A3F 自适应聚合：动态平衡先验增强特征和深层融合特征；
5. 工程化封装：模型被封装为 Qwen 大模型 Agent 可调用工具，支持图像融合、指标解释、RAG检索和报告生成。

该工具体现了“大模型理解用户任务 -> 自动选择融合工具 -> 调用本地深度学习模型 -> 返回结果 -> 大模型总结”的完整Agent流程。
"""

    return [
        wemfusion_fusion_tool,
        rag_search_tool,
        product_advice_tool,
        fault_diagnosis_tool,
        weather_maintenance_tool,
        current_fusion_metrics_tool,
        fusion_improvement_advice_tool,
        current_report_tool,
        wemfusion_model_info_tool,
    ]
