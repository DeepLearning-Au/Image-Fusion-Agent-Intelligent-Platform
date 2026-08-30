"""
MambaDFuse-Agent FastAPI backend with streaming output and workflow trace panel.

放置位置：
mambadfuse_agent_updata/backend/main.py

推荐启动：
cd "F:\\deepleaning\\融合大模型"
$env:PYTHONPATH = "F:\\deepleaning\\融合大模型;F:\\deepleaning\\融合大模型\\mambadfuse_agent_updata;" + $env:PYTHONPATH
E:\\anaconda\\envs\\mamba\\python.exe -m uvicorn mambadfuse_agent_updata.backend.main:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import json
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

from fastapi import Body, FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =============================================================================
# Path bootstrap
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]
PARENT_DIR = PROJECT_DIR.parent
for p in [str(PROJECT_DIR), str(PARENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from config import settings
except Exception as e:
    raise RuntimeError(f"无法导入 config.settings，请确认从项目上一级目录启动 uvicorn。原始错误：{e}")

TEMP_DIR = Path(getattr(settings, "TEMP_DIR", PROJECT_DIR / "data" / "temp"))
OUTPUT_DIR = Path(getattr(settings, "OUTPUT_DIR", PROJECT_DIR / "outputs"))
STORAGE_DIR = PROJECT_DIR / "storage"
WEB_DIR = PROJECT_DIR / "web"
KNOWLEDGE_DIR = PROJECT_DIR / "data" / "knowledge_docs"
KNOWLEDGE_ASSETS_DIR = PROJECT_DIR / "data" / "knowledge_assets"
PRODUCT_CATALOG_FILE = KNOWLEDGE_ASSETS_DIR / "products" / "catalog.json"
SESSIONS_FILE = STORAGE_DIR / "session_history.json"
for d in [TEMP_DIR, OUTPUT_DIR, STORAGE_DIR, KNOWLEDGE_DIR, KNOWLEDGE_ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# App
# =============================================================================
app = FastAPI(
    title="MambaDFuse-Agent API",
    description="FastAPI backend with streaming chat, fusion, RAG search, and visible agent workflow trace.",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/temp", StaticFiles(directory=str(TEMP_DIR)), name="temp")
app.mount("/knowledge-assets", StaticFiles(directory=str(KNOWLEDGE_ASSETS_DIR)), name="knowledge-assets")
if WEB_DIR.exists() and (WEB_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")

# =============================================================================
# Runtime state
# =============================================================================
SESSION_STATE: Dict[str, Any] = {
    "messages": [],
    "last_result": None,
    "last_report": None,
    "batch_results": [],
}
SESSIONS: Dict[str, Dict[str, Any]] = {}


def load_sessions() -> Dict[str, Dict[str, Any]]:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_sessions() -> None:
    payload = json.dumps(json_safe(SESSIONS), ensure_ascii=False, indent=2)
    tmp_path = SESSIONS_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(SESSIONS_FILE)


SESSIONS.update(load_sessions())

# =============================================================================
# Schemas
# =============================================================================
class ChatRequest(BaseModel):
    question: Optional[str] = None
    prompt: Optional[str] = None
    session_id: Optional[str] = None
    user_city: Optional[str] = None
    use_tile: bool = False
    tile: int = 256
    overlap: int = 32


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class VectorRebuildRequest(BaseModel):
    force: bool = False


class RerankerPreloadRequest(BaseModel):
    allow_download: bool = False


class SupplementalImportRequest(BaseModel):
    build_vectors: bool = True


class RechunkRequest(BaseModel):
    build_vectors: bool = True


class GenerationBuildRequest(BaseModel):
    name: Optional[str] = None
    publish: bool = True


class KBTextRequest(BaseModel):
    title: str
    content: str
    category: str = "手动添加"
    enabled: bool = True


class KBEnabledRequest(BaseModel):
    enabled: bool = True


class ProductCandidateReviewRequest(BaseModel):
    review_status: str
    spec_key: Optional[str] = None
    spec_label: Optional[str] = None
    category: Optional[str] = None
    value_type: Optional[str] = None
    value: Any = None
    unit: Optional[str] = None
    comparator: Optional[str] = None
    review_note: str = ""
    reviewed_by: str = "local_admin"


class ProductCandidatePublishRequest(BaseModel):
    model: str


class DeviceHybridSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class QueueProbeRequest(BaseModel):
    value: str = "ok"


class SessionMessageRequest(BaseModel):
    role: str
    content: str


class HealthResponse(BaseModel):
    ok: bool
    time: str
    project_dir: str
    python: str

# =============================================================================
# Helpers
# =============================================================================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def json_safe(obj: Any, max_list_len: int = 80) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            value = obj.tolist()
            if isinstance(value, list):
                if len(value) > max_list_len and all(isinstance(x, (int, float)) for x in value[:10]):
                    return f"<vector length={len(value)}>"
                return json_safe(value, max_list_len=max_list_len)
            return value
        except Exception:
            pass
    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            key = str(k)
            if key.lower() in ["embedding", "embeddings", "vector", "vectors", "dense_vector", "dense_vectors"]:
                try:
                    safe[key] = f"<removed vector/list length={len(v)}>"
                except Exception:
                    safe[key] = "<removed vector>"
                continue
            safe[key] = json_safe(v, max_list_len=max_list_len)
        return safe
    if isinstance(obj, (list, tuple)):
        if len(obj) > max_list_len and all(isinstance(x, (int, float)) for x in obj[:10]):
            return f"<vector length={len(obj)}>"
        return [json_safe(x, max_list_len=max_list_len) for x in obj]
    try:
        return str(obj)
    except Exception:
        return "<unserializable object>"


def ensure_session(session_id: Optional[str] = None) -> str:
    sid = session_id or uuid.uuid4().hex
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "id": sid,
            "created_at": now_str(),
            "updated_at": now_str(),
            "messages": [],
            "last_result": None,
            "last_report": None,
            "current_optical_path": None,
            "current_sar_path": None,
        }
        save_sessions()
    else:
        SESSIONS[sid]["updated_at"] = now_str()
        save_sessions()
    return sid


def append_message(session_id: str, role: str, content: str) -> None:
    ensure_session(session_id)
    SESSIONS[session_id]["messages"].append({"role": role, "content": content})
    SESSIONS[session_id]["updated_at"] = now_str()
    save_sessions()


async def save_upload_file(upload: UploadFile, prefix: str = "upload") -> Path:
    suffix = Path(upload.filename or "").suffix or ".bin"
    filename = f"{prefix}_{uuid.uuid4().hex}{suffix}"
    save_path = TEMP_DIR / filename
    content = await upload.read()
    save_path.write_bytes(content)
    return save_path


def path_to_output_url(path_value: Any) -> Optional[str]:
    if not path_value:
        return None
    p = Path(str(path_value))
    try:
        rel = p.resolve().relative_to(OUTPUT_DIR.resolve())
        return "/outputs/" + rel.as_posix()
    except Exception:
        pass
    try:
        rel = p.resolve().relative_to(TEMP_DIR.resolve())
        return "/temp/" + rel.as_posix()
    except Exception:
        pass
    return None


def enrich_fusion_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = json_safe(result or {})
    url_keys = [
        "optical_path", "sar_path", "fused_path", "fused_gray_path", "fused_color_path",
        "edge_optical_path", "edge_sar_path", "edge_fused_path", "diff_path",
    ]
    urls: Dict[str, Optional[str]] = {}
    for key in url_keys:
        if key in result and result.get(key):
            urls[key.replace("_path", "_url")] = path_to_output_url(result.get(key))
    result["urls"] = urls
    for k, v in urls.items():
        result[k] = v
    return result


def ndjson_event(event_type: str, payload: Dict[str, Any]) -> str:
    data = {"type": event_type, **json_safe(payload)}
    return json.dumps(data, ensure_ascii=False) + "\n"


def stream_text(answer: str, chunk_size: int = 24) -> Generator[str, None, None]:
    if not answer:
        return
    for i in range(0, len(answer), chunk_size):
        yield ndjson_event("answer_chunk", {"text": answer[i:i + chunk_size]})


def build_single_report_markdown(result: Dict[str, Any]) -> str:
    if not result:
        return ""
    metrics = result.get("metrics", {}) or {}
    score = float(metrics.get("综合质量评分", 0) or 0)
    qabf = float(metrics.get("Qabf 边缘保持", 0) or 0)
    mi = float(metrics.get("MI 互信息", 0) or 0)
    ssim = float(metrics.get("SSIM 结构相似性", 0) or 0)
    ag = float(metrics.get("AG 平均梯度", 0) or 0)
    level = "较好" if score >= 70 else ("中等" if score >= 55 else "偏低")
    lines = [
        "# MambaDFuse-Agent 红外—可见光图像融合分析报告",
        "",
        f"生成时间：{now_str()}",
        "",
        "## 1. 输入与输出",
        f"- 可见光图像：{result.get('optical_path', '')}",
        f"- 红外图像：{result.get('sar_path', '')}",
        f"- 融合图像：{result.get('fused_color_path', '')}",
        f"- 推理设备：{result.get('device', '')}",
        f"- 推理耗时：{float(result.get('elapsed_seconds', 0) or 0):.2f} s",
        "",
        "## 2. 核心指标",
        f"- 综合质量评分：{score:.2f}/100，整体等级：{level}",
        f"- Qabf 边缘保持：{qabf:.4f}",
        f"- MI 互信息：{mi:.4f}",
        f"- SSIM 结构相似性：{ssim:.4f}",
        f"- AG 平均梯度：{ag:.4f}",
        "",
        "## 3. 诊断结论",
    ]
    if score >= 70:
        lines.append("当前融合质量较好，可见光纹理与红外显著目标信息保留较充分。")
    elif score >= 55:
        lines.append("当前融合质量处于中等水平，具备初步判读价值，但仍建议关注边缘保持、结构一致性和配准质量。")
    else:
        lines.append("当前融合质量偏低，建议优先检查红外—可见光配准、输入图像质量和模型权重适配情况。")
    lines.extend([
        "",
        "## 4. 可能问题",
        "- 若 Qabf 偏低，通常说明边缘保持不足、配准误差或输入图像噪声干扰。",
        "- 若 MI 偏低，说明融合图像对两类源图像的信息继承仍有提升空间。",
        "- 若 SSIM 偏低，建议检查可见光图像与红外图像是否存在空间错位。",
        "",
        "## 5. 优化建议",
        "1. 融合前增加红外—可见光配准检测。",
        "2. 对噪声明显的红外图像进行与设备匹配的预处理。",
        "3. 训练阶段加入 Sobel 梯度损失或边缘保持损失。",
        "4. 对复杂场景增加雨雾、低照度、强散射样本。",
    ])
    return "\n".join(lines)


def build_batch_report_markdown(batch_results: List[Dict[str, Any]]) -> str:
    if not batch_results:
        return "# MambaDFuse-Agent 批量融合分析报告\n\n当前没有可用的批量融合结果。"

    rows = []
    scores = []
    qabfs = []
    mis = []
    ssims = []
    for item in batch_results:
        metrics = item.get("metrics", {}) or {}
        score = float(metrics.get("综合质量评分", 0) or 0)
        qabf = float(metrics.get("Qabf 边缘保持", 0) or 0)
        mi = float(metrics.get("MI 互信息", 0) or 0)
        ssim = float(metrics.get("SSIM 结构相似性", 0) or 0)
        scores.append(score)
        qabfs.append(qabf)
        mis.append(mi)
        ssims.append(ssim)
        rows.append([
            str(item.get("_pair_index", "")),
            str(item.get("_optical_name", "")),
            str(item.get("_sar_name", "")),
            f"{score:.2f}",
            f"{float(metrics.get('EN 信息熵', 0) or 0):.4f}",
            f"{float(metrics.get('MI 互信息', 0) or 0):.4f}",
            f"{float(metrics.get('SSIM 结构相似性', 0) or 0):.4f}",
            f"{float(metrics.get('Qabf 边缘保持', 0) or 0):.4f}",
        ])

    avg_score = sum(scores) / len(scores)
    avg_qabf = sum(qabfs) / len(qabfs)
    avg_mi = sum(mis) / len(mis)
    avg_ssim = sum(ssims) / len(ssims)
    best_i = max(range(len(scores)), key=lambda i: scores[i])
    worst_i = min(range(len(scores)), key=lambda i: scores[i])
    level = "较好" if avg_score >= 70 else ("中等" if avg_score >= 55 else "偏低")

    lines = [
        "# MambaDFuse-Agent 批量红外—可见光图像融合分析报告",
        "",
        f"生成时间：{now_str()}",
        f"处理数量：{len(batch_results)} 组",
        f"平均综合质量评分：{avg_score:.2f}/100，整体等级：{level}",
        "",
        "## 1. 批量指标汇总",
        "",
        "| 序号 | 可见光图像 | 红外图像 | 评分 | EN | MI | SSIM | Qabf |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend([
        "",
        "## 2. 整体结论",
        "",
        f"- 平均 Qabf：{avg_qabf:.4f}，平均 MI：{avg_mi:.4f}，平均 SSIM：{avg_ssim:.4f}。",
        f"- 表现最好：第 {batch_results[best_i].get('_pair_index')} 组，评分 {scores[best_i]:.2f}/100。",
        f"- 表现相对较弱：第 {batch_results[worst_i].get('_pair_index')} 组，评分 {scores[worst_i]:.2f}/100。",
        "",
        "## 3. 优化建议",
        "",
        "1. 优先检查可见光图像与红外图像的空间配准，避免边缘错位。",
        "2. 对噪声明显的红外图像进行与设备匹配的预处理。",
        "3. 若 Qabf 或 SSIM 偏低，建议加强边缘保持和结构一致性约束。",
        "4. 对低照度、雨雾、强散射等企业真实场景补充样本。",
    ])
    return "\n".join(lines)


def maybe_generate_report(prompt: str, session: Dict[str, Any]) -> Optional[str]:
    if not session.get("last_result"):
        return None
    keywords = ["报告", "生成报告", "分析报告", "出报告", "写报告", "下载报告"]
    if any(k in (prompt or "") for k in keywords):
        report = build_single_report_markdown(session["last_result"])
        session["last_report"] = report
        save_sessions()
        return report
    return None


def compact_trace(trace: Any) -> List[Dict[str, Any]]:
    out = []
    for i, step in enumerate(trace or [], start=1):
        if isinstance(step, dict):
            out.append({
                "index": i,
                "tool": step.get("tool") or step.get("name") or step.get("tool_name") or f"step_{i}",
                "input": json_safe(step.get("tool_input") or step.get("input") or step.get("args") or ""),
                "output": json_safe(step.get("observation") or step.get("output") or step.get("result") or ""),
            })
        else:
            out.append({"index": i, "tool": f"step_{i}", "input": "", "output": json_safe(step)})
    return out


# =============================================================================
# SQLite KB import helper
# =============================================================================
def kb_service():
    try:
        from rag import sqlite_kb
        sqlite_kb.init_db()
        return sqlite_kb
    except Exception as e:
        raise RuntimeError(f"无法加载 SQLite 知识库模块 rag/sqlite_kb.py：{e}")

# =============================================================================
# Frontend and system routes
# =============================================================================
@app.get("/", include_in_schema=False)
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"message": "FastAPI backend is running, but web/index.html was not found.", "docs": "/docs", "kb_admin": "/kb"})
    return FileResponse(str(index_path), headers={"Cache-Control": "no-store"})


@app.get("/kb", include_in_schema=False)
def kb_page():
    kb_path = WEB_DIR / "kb.html"
    if not kb_path.exists():
        return JSONResponse({"message": "KB admin page not found: web/kb.html", "docs": "/docs"})
    return FileResponse(str(kb_path), headers={"Cache-Control": "no-store"})


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(ok=True, time=now_str(), project_dir=str(PROJECT_DIR), python=sys.executable)


@app.get("/api/system/info")
def system_info():
    return JSONResponse(content=json_safe({
        "project_dir": str(PROJECT_DIR),
        "parent_dir": str(PARENT_DIR),
        "python": sys.executable,
        "temp_dir": str(TEMP_DIR),
        "output_dir": str(OUTPUT_DIR),
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "session_file": str(SESSIONS_FILE),
        "session_file_exists": SESSIONS_FILE.exists(),
        "time": now_str(),
        "session": {
            "messages": len(SESSION_STATE.get("messages", [])),
            "has_last_result": SESSION_STATE.get("last_result") is not None,
            "batch_count": len(SESSION_STATE.get("batch_results", []) or []),
            "has_last_report": bool(SESSION_STATE.get("last_report")),
            "named_sessions": len(SESSIONS),
        },
    }))

# =============================================================================
# Streaming chat
# =============================================================================
@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    question = (req.prompt or req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question/prompt 不能为空")
    session_id = ensure_session(req.session_id)
    session = SESSIONS[session_id]
    append_message(session_id, "user", question)

    def gen():
        try:
            yield ndjson_event("workflow", {"stage": "接收问题", "detail": question})
            yield ndjson_event("workflow", {"stage": "意图识别", "detail": "未上传图片，进入普通问答 / Agent 工具调度流程"})
            yield ndjson_event("workflow", {"stage": "加载上下文", "detail": "读取当前会话历史、最近融合结果和报告状态"})

            report_text = maybe_generate_report(question, session)
            if report_text:
                answer = "已根据当前融合结果生成分析报告。"
                append_message(session_id, "assistant", answer)
                yield ndjson_event("workflow", {"stage": "报告生成", "detail": "根据当前 last_result 生成 Markdown 分析报告", "tool": "build_single_report_markdown"})
                for chunk in stream_text(answer):
                    yield chunk
                yield ndjson_event("final", {"answer": answer, "report_text": report_text, "trace": [{"tool": "build_single_report_markdown", "input": question, "output": "report_text"}]})
                return

            yield ndjson_event("workflow", {"stage": "调用大模型 Agent", "detail": "调用 run_langchain_agent，模型会按需选择 RAG、天气、指标解释等工具", "tool": "run_langchain_agent"})
            from agent.react_agent import run_langchain_agent

            output = run_langchain_agent(
                question=question,
                history=session.get("messages", [])[:-1],
                last_result=session.get("last_result") or SESSION_STATE.get("last_result"),
                last_report=session.get("last_report") or SESSION_STATE.get("last_report"),
                optical_path=session.get("current_optical_path"),
                sar_path=session.get("current_sar_path"),
                use_tile=req.use_tile,
                tile=req.tile,
                overlap=req.overlap,
                user_city=req.user_city or getattr(settings, "AGENT_USER_CITY", "烟台"),
            )

            tool_state = output.get("tool_state", {}) or {}
            if tool_state.get("last_result") is not None:
                result = enrich_fusion_result(tool_state.get("last_result"))
                session["last_result"] = result
                SESSION_STATE["last_result"] = result
            if tool_state.get("last_report") is not None:
                session["last_report"] = tool_state.get("last_report")
                SESSION_STATE["last_report"] = tool_state.get("last_report")
            save_sessions()

            answer = output.get("answer", "")
            trace = compact_trace(output.get("trace", []))
            append_message(session_id, "assistant", answer)
            yield ndjson_event("workflow", {"stage": "整理输出", "detail": f"Agent 返回 {len(trace)} 条工具调用轨迹"})
            yield ndjson_event("trace", {"trace": trace})
            for chunk in stream_text(answer):
                yield chunk
            yield ndjson_event("final", {
                "answer": answer,
                "trace": trace,
                "fusion_result": session.get("last_result") if tool_state.get("last_result") is not None else None,
                "report_text": tool_state.get("last_report"),
            })
        except Exception as e:
            yield ndjson_event("error", {"message": f"Agent 调用失败：{e}", "traceback": traceback.format_exc()})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/chat")
def chat(req: ChatRequest):
    # 非流式兼容接口
    question = (req.prompt or req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question/prompt 不能为空")
    session_id = ensure_session(req.session_id)
    session = SESSIONS[session_id]
    append_message(session_id, "user", question)
    try:
        from agent.react_agent import run_langchain_agent
        output = run_langchain_agent(
            question=question,
            history=session.get("messages", [])[:-1],
            last_result=session.get("last_result") or SESSION_STATE.get("last_result"),
            last_report=session.get("last_report") or SESSION_STATE.get("last_report"),
            optical_path=session.get("current_optical_path"),
            sar_path=session.get("current_sar_path"),
            use_tile=req.use_tile,
            tile=req.tile,
            overlap=req.overlap,
            user_city=req.user_city or getattr(settings, "AGENT_USER_CITY", "烟台"),
        )
        answer = output.get("answer", "")
        trace = compact_trace(output.get("trace", []))
        append_message(session_id, "assistant", answer)
        return JSONResponse(content=json_safe({"answer": answer, "trace": trace}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({"detail": f"Agent 调用失败：{e}", "traceback": traceback.format_exc()}))

# =============================================================================
# Streaming fusion
# =============================================================================
@app.post("/api/chat/fusion/stream")
async def chat_fusion_stream(
    prompt: str = Form("请融合我上传的可见光图像和红外图像，并给出简要分析。"),
    session_id: Optional[str] = Form(None),
    generate_report: bool = Form(False),
    use_tile: bool = Form(False),
    tile: int = Form(256),
    overlap: int = Form(32),
    optical_file: Optional[UploadFile] = File(None),
    sar_file: Optional[UploadFile] = File(None),
):
    session_id = ensure_session(session_id)
    session = SESSIONS[session_id]
    append_message(session_id, "user", prompt)

    if optical_file is None or sar_file is None:
        raise HTTPException(status_code=400, detail="请同时上传可见光图像和红外图像。")

    optical_path = await save_upload_file(optical_file, "chat_optical")
    sar_path = await save_upload_file(sar_file, "chat_sar")
    session["current_optical_path"] = str(optical_path)
    session["current_sar_path"] = str(sar_path)
    save_sessions()

    def gen():
        try:
            yield ndjson_event("workflow", {"stage": "接收图片", "detail": f"可见光图：{optical_file.filename}；红外图：{sar_file.filename}"})
            yield ndjson_event("workflow", {"stage": "保存临时文件", "detail": f"{optical_path.name} / {sar_path.name}"})
            yield ndjson_event("workflow", {"stage": "调用本地融合模型", "detail": "调用 MambaDFuse 进行红外—可见光融合", "tool": "run_wemfusion"})

            from fusion.wemfusion_runner import run_wemfusion
            result = run_wemfusion(
                optical_path=str(optical_path),
                sar_path=str(sar_path),
                output_dir=OUTPUT_DIR,
                use_tile=use_tile,
                tile=tile,
                overlap=overlap,
            )
            result = enrich_fusion_result(result)
            result["_optical_name"] = optical_file.filename
            result["_sar_name"] = sar_file.filename
            session["last_result"] = result
            SESSION_STATE["last_result"] = result
            save_sessions()

            yield ndjson_event("workflow", {"stage": "计算融合指标", "detail": "读取 EN、MI、SSIM、Qabf、综合质量评分等指标", "tool": "fusion.metrics"})
            report_text = build_single_report_markdown(result) if generate_report else None
            if report_text:
                session["last_report"] = report_text
                SESSION_STATE["last_report"] = report_text
                save_sessions()
                yield ndjson_event("workflow", {"stage": "生成分析报告", "detail": "根据当前融合结果生成 Markdown 报告", "tool": "build_single_report_markdown"})

            metrics = result.get("metrics", {}) or {}
            answer = (
                f"已完成红外—可见光图像融合。综合质量评分：{float(metrics.get('综合质量评分', 0) or 0):.2f}/100；"
                f"Qabf：{float(metrics.get('Qabf 边缘保持', 0) or 0):.4f}；"
                f"MI：{float(metrics.get('MI 互信息', 0) or 0):.4f}；"
                f"SSIM：{float(metrics.get('SSIM 结构相似性', 0) or 0):.4f}。"
            )
            if report_text:
                answer += " 已同步生成报告。"
            else:
                answer += " 你可以继续追问，例如“Qabf 为什么不高？”或“生成报告”。"

            trace = [
                {"tool": "save_upload_file", "input": "optical_file, sar_file", "output": "temporary image paths"},
                {"tool": "run_wemfusion", "input": {"use_tile": use_tile, "tile": tile, "overlap": overlap}, "output": "fusion result + metrics"},
            ]
            if report_text:
                trace.append({"tool": "build_single_report_markdown", "input": "fusion_result", "output": "report_text"})

            append_message(session_id, "assistant", answer)
            yield ndjson_event("trace", {"trace": trace})
            yield ndjson_event("fusion_result", {"fusion_result": result, "report_text": report_text})
            for chunk in stream_text(answer):
                yield chunk
            yield ndjson_event("final", {"answer": answer, "fusion_result": result, "report_text": report_text, "trace": trace})
        except Exception as e:
            yield ndjson_event("error", {"message": f"融合失败：{e}", "traceback": traceback.format_exc()})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/chat/fusion")
async def chat_fusion(
    prompt: str = Form("请融合我上传的可见光图像和红外图像，并给出简要分析。"),
    session_id: Optional[str] = Form(None),
    generate_report: bool = Form(False),
    use_tile: bool = Form(False),
    tile: int = Form(256),
    overlap: int = Form(32),
    optical_file: Optional[UploadFile] = File(None),
    sar_file: Optional[UploadFile] = File(None),
):
    if optical_file is None or sar_file is None:
        raise HTTPException(status_code=400, detail="请同时上传可见光图像和红外图像。")
    try:
        optical_path = await save_upload_file(optical_file, "chat_optical")
        sar_path = await save_upload_file(sar_file, "chat_sar")
        from fusion.wemfusion_runner import run_wemfusion
        result = run_wemfusion(str(optical_path), str(sar_path), output_dir=OUTPUT_DIR, use_tile=use_tile, tile=tile, overlap=overlap)
        result = enrich_fusion_result(result)
        report_text = build_single_report_markdown(result) if generate_report else None
        return JSONResponse(content=json_safe({"answer": "融合完成。", "fusion_result": result, "report_text": report_text, "trace": []}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({"detail": f"融合失败：{e}", "traceback": traceback.format_exc()}))


@app.post("/api/fusion/batch/stream")
async def batch_fusion_stream(
    session_id: Optional[str] = Form(None),
    generate_report: bool = Form(False),
    use_tile: bool = Form(False),
    tile: int = Form(256),
    overlap: int = Form(32),
    pair_mode: str = Form("name"),
    optical_files: List[UploadFile] = File(...),
    sar_files: List[UploadFile] = File(...),
):
    if not optical_files or not sar_files:
        raise HTTPException(status_code=400, detail="请批量上传可见光图像和红外图像。")

    session_id = ensure_session(session_id)
    session = SESSIONS[session_id]

    optical_items = []
    sar_items = []
    for file in optical_files:
        optical_items.append({"name": file.filename or "optical", "path": await save_upload_file(file, "batch_optical")})
    for file in sar_files:
        sar_items.append({"name": file.filename or "sar", "path": await save_upload_file(file, "batch_sar")})

    if pair_mode == "name":
        optical_items = sorted(optical_items, key=lambda x: x["name"])
        sar_items = sorted(sar_items, key=lambda x: x["name"])

    pair_count = min(len(optical_items), len(sar_items))
    if pair_count <= 0:
        raise HTTPException(status_code=400, detail="没有可配对的图像。")

    def gen():
        batch_results: List[Dict[str, Any]] = []
        try:
            yield ndjson_event("workflow", {
                "stage": "批量接收图片",
                "detail": f"可见光图像 {len(optical_items)} 张，红外图像 {len(sar_items)} 张，可处理 {pair_count} 组",
            })
            if len(optical_items) != len(sar_items):
                yield ndjson_event("workflow", {
                    "stage": "数量不一致",
                    "detail": f"本次只处理前 {pair_count} 组，多出的图像会被跳过。",
                })

            from fusion.wemfusion_runner import run_wemfusion

            for i in range(pair_count):
                optical_item = optical_items[i]
                sar_item = sar_items[i]
                yield ndjson_event("workflow", {
                    "stage": f"处理第 {i + 1}/{pair_count} 组",
                    "detail": f"{optical_item['name']} + {sar_item['name']}",
                    "tool": "run_wemfusion",
                })

                result = run_wemfusion(
                    optical_path=str(optical_item["path"]),
                    sar_path=str(sar_item["path"]),
                    output_dir=OUTPUT_DIR,
                    use_tile=use_tile,
                    tile=tile,
                    overlap=overlap,
                )
                result = enrich_fusion_result(result)
                result["_pair_index"] = i + 1
                result["_optical_name"] = optical_item["name"]
                result["_sar_name"] = sar_item["name"]
                batch_results.append(result)

                yield ndjson_event("batch_item", {
                    "index": i + 1,
                    "total": pair_count,
                    "fusion_result": result,
                    "progress": round((i + 1) / pair_count, 4),
                })

            SESSION_STATE["batch_results"] = batch_results
            SESSION_STATE["last_result"] = batch_results[-1] if batch_results else None
            session["last_result"] = SESSION_STATE["last_result"]
            save_sessions()

            report_text = build_batch_report_markdown(batch_results) if generate_report else None
            if report_text:
                SESSION_STATE["last_report"] = report_text
                session["last_report"] = report_text
                save_sessions()

            answer = f"批量融合完成：成功处理 {len(batch_results)}/{pair_count} 组图像。"
            if report_text:
                answer += " 已同步生成批量分析报告。"
            append_message(session_id, "assistant", answer)

            yield ndjson_event("workflow", {"stage": "批量融合完成", "detail": answer})
            yield ndjson_event("final", {
                "answer": answer,
                "batch_results": batch_results,
                "report_text": report_text,
                "trace": [{"tool": "run_wemfusion", "input": {"pair_count": pair_count, "use_tile": use_tile, "tile": tile, "overlap": overlap}, "output": "batch fusion results"}],
            })
        except Exception as e:
            yield ndjson_event("error", {"message": f"批量融合失败：{e}", "traceback": traceback.format_exc()})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# =============================================================================
# Fusion current / report
# =============================================================================
@app.get("/api/fusion/current")
def fusion_current():
    return JSONResponse(content=json_safe({"last_result": SESSION_STATE.get("last_result"), "batch_results": SESSION_STATE.get("batch_results", [])}))


# =============================================================================
# Redis / Celery async tasks
# =============================================================================
def _task_response(task_id: str, task_type: str) -> JSONResponse:
    location = f"/api/tasks/{task_id}"
    return JSONResponse(
        status_code=202,
        headers={"Location": location},
        content={
            "success": True,
            "accepted": True,
            "task_id": task_id,
            "task_type": task_type,
            "status": "queued",
            "status_url": location,
        },
    )


def _require_task_queue() -> None:
    if not settings.TASK_QUEUE_ENABLED:
        raise HTTPException(status_code=503, detail="Redis任务队列当前未启用")


@app.get("/api/tasks/status")
def task_queue_status():
    from task_queue.service import queue_status

    result = queue_status()
    status_code = 200 if result.get("redis_ready") else 503
    return JSONResponse(status_code=status_code, content=json_safe({"success": status_code == 200, **result}))


@app.post("/api/tasks/probe")
def task_queue_probe(req: QueueProbeRequest):
    _require_task_queue()
    try:
        from task_queue.service import enqueue_probe

        return _task_response(enqueue_probe(req.value), "probe")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"任务入队失败：{exc}") from exc


@app.post("/api/tasks/fusion")
async def enqueue_fusion_api(
    prompt: str = Form("请融合可见光图像和红外图像"),
    generate_report: bool = Form(False),
    use_tile: bool = Form(False),
    tile: int = Form(256),
    overlap: int = Form(32),
    optical_file: UploadFile = File(...),
    infrared_file: UploadFile = File(...),
):
    _require_task_queue()
    optical_path = await save_upload_file(optical_file, "queue_optical")
    infrared_path = await save_upload_file(infrared_file, "queue_infrared")
    try:
        from task_queue.service import enqueue_fusion

        task_id = enqueue_fusion(
            optical_path=str(optical_path),
            infrared_path=str(infrared_path),
            use_tile=use_tile,
            tile=tile,
            overlap=overlap,
            generate_report=generate_report,
            user_note=prompt,
        )
        return _task_response(task_id, "fusion")
    except Exception as exc:
        optical_path.unlink(missing_ok=True)
        infrared_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"融合任务入队失败：{exc}") from exc


@app.post("/api/tasks/fusion/batch")
async def enqueue_batch_fusion_api(
    generate_report: bool = Form(False),
    use_tile: bool = Form(False),
    tile: int = Form(256),
    overlap: int = Form(32),
    pair_mode: str = Form("name"),
    optical_files: List[UploadFile] = File(...),
    sar_files: List[UploadFile] = File(...),
):
    _require_task_queue()
    if not optical_files or not sar_files:
        raise HTTPException(status_code=400, detail="请批量上传可见光图像和红外图像")

    optical_items: List[Dict[str, str]] = []
    infrared_items: List[Dict[str, str]] = []
    try:
        for upload in optical_files:
            path = await save_upload_file(upload, "queue_batch_optical")
            optical_items.append({"name": upload.filename or path.name, "path": str(path)})
        for upload in sar_files:
            path = await save_upload_file(upload, "queue_batch_infrared")
            infrared_items.append({"name": upload.filename or path.name, "path": str(path)})

        if pair_mode == "name":
            optical_items.sort(key=lambda item: item["name"])
            infrared_items.sort(key=lambda item: item["name"])

        from task_queue.service import enqueue_batch_fusion

        task_id = enqueue_batch_fusion(
            optical_items=optical_items,
            infrared_items=infrared_items,
            use_tile=use_tile,
            tile=tile,
            overlap=overlap,
            generate_report=generate_report,
        )
        return _task_response(task_id, "batch_fusion")
    except Exception as exc:
        for item in optical_items + infrared_items:
            Path(item["path"]).unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"批量融合任务入队失败：{exc}") from exc


@app.post("/api/tasks/rag/generation")
def enqueue_rag_generation_api(req: GenerationBuildRequest):
    _require_task_queue()
    try:
        from task_queue.service import enqueue_generation

        return _task_response(
            enqueue_generation(name=req.name, publish=req.publish),
            "rag_generation",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"知识库构建任务入队失败：{exc}") from exc


@app.post("/api/tasks/rag/vector-sync")
def enqueue_vector_sync_api(req: VectorRebuildRequest):
    _require_task_queue()
    try:
        from task_queue.service import enqueue_vector_sync

        return _task_response(enqueue_vector_sync(force=req.force), "vector_sync")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"向量同步任务入队失败：{exc}") from exc


@app.get("/api/tasks/{task_id}")
def task_result_api(task_id: str):
    _require_task_queue()
    try:
        uuid.UUID(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="task_id格式不正确") from exc
    from task_queue.service import get_task_status

    return JSONResponse(content=json_safe({"success": True, **get_task_status(task_id)}))


@app.post("/api/tasks/{task_id}/cancel")
def task_cancel_api(task_id: str):
    _require_task_queue()
    try:
        uuid.UUID(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="task_id格式不正确") from exc
    from task_queue.service import get_task_status, revoke_task

    status = get_task_status(task_id)
    if status.get("ready"):
        raise HTTPException(status_code=409, detail="任务已经结束，不能取消")
    return JSONResponse(content=json_safe({"success": True, **revoke_task(task_id)}))


@app.post("/api/report/generate")
def report_generate():
    last_result = SESSION_STATE.get("last_result")
    batch_results = SESSION_STATE.get("batch_results") or []
    if batch_results:
        report = build_batch_report_markdown(batch_results)
        SESSION_STATE["last_report"] = report
        return JSONResponse(content=json_safe({"ok": True, "report": report, "type": "batch"}))
    if not last_result:
        raise HTTPException(status_code=400, detail="当前没有融合结果，无法生成报告")
    report = build_single_report_markdown(last_result)
    SESSION_STATE["last_report"] = report
    return JSONResponse(content=json_safe({"ok": True, "report": report, "type": "single"}))

# =============================================================================
# RAG / KB APIs
# =============================================================================
@app.get("/api/rag/status")
def rag_status():
    try:
        kb = kb_service()
        return JSONResponse(content=json_safe({"success": True, **kb.get_status()}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({"success": False, "detail": str(e), "traceback": traceback.format_exc()}))


@app.post("/api/rag/search")
def rag_search(req: RAGSearchRequest):
    try:
        from rag.enterprise_retrieval import enterprise_search
        result = enterprise_search(req.query, top_k=req.top_k)
        return JSONResponse(content=json_safe({
            "success": True,
            "query": req.query,
            "confidence": result.get("confidence", "low"),
            "confidence_reason": result.get("confidence_reason", ""),
            "top_score": result.get("top_score", 0),
            "source_count": len(result.get("sources", [])),
            "sources": result.get("sources", []),
            "rewritten_queries": [req.query],
            "context": result.get("context", ""),
            "source_text": result.get("source_text", ""),
            "retrieval": result.get("retrieval", {}),
            "warnings": result.get("warnings", []),
        }))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({"success": False, "message": f"RAG 检索失败：{e}", "traceback": traceback.format_exc()}))


@app.post("/api/rag/rebuild")
def rag_rebuild():
    try:
        from rag.generation_service import build_generation

        result = build_generation(publish=True)
        return JSONResponse(content=json_safe({"message": "知识库新版本构建并发布完成", **result}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({"success": False, "message": f"知识库重建失败：{e}", "traceback": traceback.format_exc()}))


@app.post("/api/rag/import-supplemental")
def rag_import_supplemental(req: SupplementalImportRequest):
    try:
        from rag.generation_service import build_generation

        result = build_generation(publish=True)
        return JSONResponse(content=json_safe({"success": True, **result}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({
            "success": False,
            "message": f"补充知识导入失败：{e}",
        }))


@app.post("/api/rag/rechunk")
def rag_rechunk(req: RechunkRequest):
    try:
        from rag.generation_service import build_generation

        result = build_generation(publish=True)
        return JSONResponse(content=json_safe({"success": True, **result}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({
            "success": False,
            "message": f"父子分块重建失败：{e}",
        }))


@app.get("/api/rag/generations")
def rag_generations():
    from rag.generation_service import list_generations

    return JSONResponse(content=json_safe({"success": True, "generations": list_generations()}))


@app.post("/api/rag/generations/build")
def rag_generation_build(req: GenerationBuildRequest):
    try:
        from rag.generation_service import build_generation

        result = build_generation(name=req.name, publish=req.publish)
        return JSONResponse(content=json_safe({"success": True, **result}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({
            "success": False,
            "message": f"generation 构建失败：{e}",
        }))


@app.post("/api/rag/generations/{generation_id}/publish")
def rag_generation_publish(generation_id: int):
    try:
        from rag.generation_service import publish_generation

        return JSONResponse(content=json_safe({
            "success": True,
            **publish_generation(generation_id),
        }))
    except Exception as e:
        return JSONResponse(status_code=400, content=json_safe({
            "success": False,
            "message": f"generation 发布失败：{e}",
        }))


@app.get("/api/rag/generations/{generation_id}/evaluation")
def rag_generation_evaluation(generation_id: int):
    from rag.generation_service import get_generation_evaluation

    result = get_generation_evaluation(generation_id)
    if not result:
        return JSONResponse(status_code=404, content={
            "success": False,
            "message": "该 generation 暂无评测报告",
        })
    return JSONResponse(content=json_safe({"success": True, **result}))


@app.get("/api/rag/generations/{generation_id}/answer-evaluation")
def rag_generation_answer_evaluation(generation_id: int):
    from rag.generation_service import get_generation_answer_evaluation

    result = get_generation_answer_evaluation(generation_id)
    if not result:
        return JSONResponse(status_code=404, content={
            "success": False,
            "message": "该 generation 暂无回答质量报告",
        })
    return JSONResponse(content=json_safe({"success": True, **result}))


@app.post("/api/rag/generations/{generation_id}/rollback")
def rag_generation_rollback(generation_id: int):
    try:
        from rag.generation_service import rollback_generation

        return JSONResponse(content=json_safe({
            "success": True,
            **rollback_generation(generation_id),
        }))
    except Exception as e:
        return JSONResponse(status_code=400, content=json_safe({
            "success": False,
            "message": f"generation 回滚失败：{e}",
        }))


@app.post("/api/rag/vector/rebuild")
def rag_vector_rebuild(req: VectorRebuildRequest):
    try:
        from rag.vector_store import sync_chunk_embeddings

        result = sync_chunk_embeddings(force=req.force)
        return JSONResponse(content=json_safe({"success": True, **result}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({
            "success": False,
            "message": f"向量索引构建失败：{e}",
        }))


@app.get("/api/rag/vector/status")
def rag_vector_status():
    try:
        from rag.vector_store import get_vector_status

        return JSONResponse(content=json_safe({"success": True, **get_vector_status()}))
    except Exception as e:
        return JSONResponse(status_code=503, content=json_safe({
            "success": False,
            "message": f"向量后端不可用：{e}",
        }))


@app.post("/api/rag/vector/index/rebuild")
def rag_vector_index_rebuild():
    try:
        if str(settings.VECTOR_STORE_BACKEND).lower() != "milvus":
            raise ValueError("当前向量后端不是Milvus")
        from rag.milvus_vector_index import rebuild_vector_index

        return JSONResponse(content=json_safe({"success": True, **rebuild_vector_index()}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({
            "success": False,
            "message": f"Milvus索引重建失败：{e}",
        }))


@app.get("/api/rag/reranker/status")
def rag_reranker_status():
    from rag.bge_reranker import reranker_status

    return JSONResponse(content=json_safe({"success": True, **reranker_status()}))


@app.post("/api/rag/reranker/preload")
def rag_reranker_preload(req: RerankerPreloadRequest):
    try:
        from rag.bge_reranker import preload_reranker

        result = preload_reranker(allow_download=req.allow_download)
        return JSONResponse(content=json_safe({"success": True, **result}))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({
            "success": False,
            "message": f"BGE Reranker 加载失败：{e}",
        }))


@app.post("/api/rag/eval")
def rag_eval():
    try:
        from rag.enterprise_retrieval import enterprise_search
        from rag.eval_service import evaluate_retrieval, load_eval_set

        eval_path = settings.RAG_EVAL_SET_PATH
        eval_items = load_eval_set(eval_path)
        result = evaluate_retrieval(
            eval_items,
            search_fn=lambda question: enterprise_search(
                question, top_k=settings.RAG_EVAL_TOP_K
            ),
            top_k=settings.RAG_EVAL_TOP_K,
        )
        return JSONResponse(content=json_safe({
            "success": True,
            "eval_set": str(eval_path),
            "top_k": settings.RAG_EVAL_TOP_K,
            **result,
        }))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({"success": False, "message": f"RAG 评估失败：{e}", "traceback": traceback.format_exc()}))


@app.post("/api/rag/answer-eval")
def rag_answer_eval():
    try:
        from rag import sqlite_kb
        from rag.generation_service import evaluate_generation_answers

        generation_id = sqlite_kb.get_active_generation_id()
        result = evaluate_generation_answers(generation_id)
        return JSONResponse(content=json_safe({
            "success": True,
            "generation_id": generation_id,
            **result,
        }))
    except Exception as e:
        return JSONResponse(status_code=500, content=json_safe({
            "success": False,
            "message": f"回答质量评估失败：{e}",
            "traceback": traceback.format_exc(),
        }))


def normalize_kb_status(status: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(status or {})
    payload.setdefault("doc_count", 0)
    payload.setdefault("enabled_doc_count", 0)
    payload.setdefault("chunk_count", 0)
    payload.setdefault("parent_chunk_count", 0)
    payload.setdefault("needs_ocr_page_count", 0)
    payload.setdefault("parse_warning_count", 0)
    payload.setdefault("table_count", 0)
    payload.setdefault("extracted_image_count", 0)
    payload.setdefault("fts5_available", False)
    payload.setdefault("embedding_count", 0)
    payload.setdefault("embedding_models", [])
    return payload


def normalize_kb_document(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row or {})
    item["doc_id"] = item.get("doc_id") or item.get("id")
    item["enabled"] = bool(item.get("enabled", True))
    item.setdefault("category", "")
    item.setdefault("source", "")
    item.setdefault("chunk_count", 0)
    return item


def normalize_kb_chunk(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row or {})
    item["chunk_id"] = item.get("chunk_id") or item.get("id")
    item["text"] = item.get("text") or item.get("content") or ""
    item["doc_title"] = item.get("doc_title") or item.get("title") or ""
    item["enabled"] = bool(item.get("enabled", True))
    return item


def kb_search_payload(query: str, top_k: int = 5) -> Dict[str, Any]:
    from rag.enterprise_retrieval import enterprise_search
    result = enterprise_search(query, top_k=top_k)
    sources = [normalize_kb_chunk(s) for s in result.get("sources", [])]
    return {
        "success": True,
        "query": query,
        "confidence": result.get("confidence", "low"),
        "confidence_reason": result.get("confidence_reason", ""),
        "top_score": result.get("top_score", 0),
        "source_count": len(sources),
        "sources": sources,
        "context": result.get("context", ""),
        "source_text": result.get("source_text", ""),
        "retrieval": result.get("retrieval", {}),
        "warnings": result.get("warnings", []),
    }


@app.get("/api/kb/status")
def kb_status():
    kb = kb_service()
    status = normalize_kb_status(kb.get_status())
    return JSONResponse(content=json_safe({"success": True, "status": status, **status}))


@app.get("/api/kb/product-assets")
def kb_product_assets():
    """返回结构化产品卡片；旧 catalog.json 只作为兼容回退。"""
    try:
        from catalog.repository import list_product_cards

        products = list_product_cards()
        if products:
            return JSONResponse(content=json_safe({
                "success": True,
                "catalog_version": "structured-v1",
                "products": products,
                "disclaimer": "产品字段来自官方资料并保留页码；未公开价格统一显示为需询价。",
            }))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"结构化产品目录读取失败：{e}")

    if PRODUCT_CATALOG_FILE.exists():
        try:
            catalog = json.loads(PRODUCT_CATALOG_FILE.read_text(encoding="utf-8"))
            return JSONResponse(content=json_safe({"success": True, **catalog}))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"旧产品图谱读取失败：{e}")
    return JSONResponse(content={"success": True, "catalog_version": "structured-v1", "products": [], "disclaimer": ""})


@app.get("/api/products/status")
def product_catalog_status():
    from catalog.repository import get_catalog_status

    return JSONResponse(content=json_safe({"success": True, **get_catalog_status()}))


@app.get("/api/product-candidates/status")
def product_candidate_status():
    from catalog.candidates import get_candidate_status

    return JSONResponse(content=json_safe({"success": True, **get_candidate_status()}))


@app.post("/api/product-candidates/generate")
def product_candidate_generate():
    from catalog.candidates import generate_candidates_from_document_tables

    result = generate_candidates_from_document_tables()
    if result["tables_seen"] == 0:
        raise HTTPException(status_code=400, detail="数据库中没有PDF表格，请先重建知识库")
    return JSONResponse(content=json_safe({"success": True, **result}))


@app.get("/api/product-candidates")
def product_candidate_list(status: Optional[str] = None, model: Optional[str] = None):
    from catalog.candidates import list_candidates

    status = status or None
    model = model or None
    allowed = {None, "pending", "approved", "rejected", "published"}
    if status not in allowed:
        raise HTTPException(status_code=400, detail="候选状态不合法")
    return JSONResponse(content=json_safe({
        "success": True,
        "candidates": list_candidates(status=status, model=model),
    }))


@app.post("/api/product-candidates/{candidate_id}/review")
def product_candidate_review(candidate_id: int, request: ProductCandidateReviewRequest):
    from catalog.candidates import review_candidate

    updates = {
        "spec_key": request.spec_key,
        "spec_label": request.spec_label,
        "category": request.category,
        "value_type": request.value_type,
        "unit": request.unit,
        "comparator": request.comparator,
    }
    provided_fields = getattr(
        request,
        "model_fields_set",
        getattr(request, "__fields_set__", set()),
    )
    if "value" in provided_fields:
        updates["value"] = request.value
    try:
        candidate = review_candidate(
            candidate_id,
            request.review_status,
            updates=updates,
            reviewed_by=request.reviewed_by,
            review_note=request.review_note,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=json_safe({"success": True, "candidate": candidate}))


@app.post("/api/product-candidates/publish")
def product_candidate_publish(request: ProductCandidatePublishRequest):
    from catalog.candidates import publish_approved_candidates

    try:
        result = publish_approved_candidates(request.model.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=json_safe({"success": True, **result}))


@app.get("/api/products")
def product_catalog_list(modality: Optional[str] = None, include_retired: bool = False):
    from catalog.repository import list_products

    products = list_products(modality=modality, include_retired=include_retired)
    return JSONResponse(content=json_safe({"success": True, "products": products}))


@app.get("/api/products/search")
def product_catalog_search(
    modality: Optional[str] = None,
    min_width: Optional[float] = None,
    min_height: Optional[float] = None,
    min_fps: Optional[float] = None,
    sdk_required: Optional[bool] = None,
):
    from catalog.repository import search_products

    products = search_products(
        modality=modality,
        min_width=min_width,
        min_height=min_height,
        min_fps=min_fps,
        sdk_required=sdk_required,
    )
    return JSONResponse(content=json_safe({
        "success": True,
        "filters": {
            "modality": modality,
            "min_width": min_width,
            "min_height": min_height,
            "min_fps": min_fps,
            "sdk_required": sdk_required,
        },
        "products": products,
    }))


@app.post("/api/products/hybrid-search")
def product_hybrid_search(request: DeviceHybridSearchRequest):
    from rag.hybrid_product_search import hybrid_product_search

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="查询内容不能为空")
    top_k = max(1, min(int(request.top_k), 10))
    return JSONResponse(content=json_safe({
        "success": True,
        **hybrid_product_search(query, top_k=top_k),
    }))


@app.get("/api/products/{model}")
def product_catalog_detail(model: str):
    from catalog.repository import get_product

    product = get_product(model)
    if not product:
        raise HTTPException(status_code=404, detail=f"未找到产品型号：{model}")
    return JSONResponse(content=json_safe({"success": True, "product": product}))


@app.get("/api/kb/documents")
def kb_documents(keyword: str = "", include_disabled: bool = True):
    kb = kb_service()
    docs = [normalize_kb_document(d) for d in kb.list_documents()]
    if not include_disabled:
        docs = [d for d in docs if d.get("enabled", True)]
    keyword = (keyword or "").strip().lower()
    if keyword:
        docs = [
            d for d in docs
            if keyword in str(d.get("title", "")).lower()
            or keyword in str(d.get("category", "")).lower()
            or keyword in str(d.get("source", "")).lower()
        ]
    return JSONResponse(content=json_safe({"success": True, "documents": docs, "status": normalize_kb_status(kb.get_status())}))


@app.get("/api/kb/chunks")
def kb_chunks(doc_id: Optional[int] = None, limit: int = 200):
    kb = kb_service()
    chunks = [normalize_kb_chunk(c) for c in kb.list_chunks(doc_id=doc_id, limit=limit)]
    return JSONResponse(content=json_safe({"success": True, "chunks": chunks}))


@app.get("/api/kb/documents/{doc_id}/parse-report")
def kb_document_parse_report(doc_id: int):
    kb = kb_service()
    report = kb.get_document_parse_report(doc_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"未找到知识库文档：{doc_id}")
    return JSONResponse(content=json_safe({"success": True, "report": report}))


@app.post("/api/kb/search")
def kb_search(req: RAGSearchRequest):
    return JSONResponse(content=json_safe(kb_search_payload(req.query, top_k=req.top_k)))


@app.post("/api/kb/add-text")
def kb_add_text(req: KBTextRequest):
    kb = kb_service()
    doc_id = kb.add_text_document(req.title, req.content, category=req.category, source="manual")
    if not req.enabled:
        kb.set_document_enabled(doc_id, False)
    return JSONResponse(content=json_safe({"success": True, "doc_id": doc_id, "status": normalize_kb_status(kb.get_status())}))


@app.post("/api/kb/upload")
async def kb_upload(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    category: str = Form("上传文档"),
    enabled: bool = Form(True),
):
    kb = kb_service()
    uploads: List[UploadFile] = []
    if files:
        uploads.extend(files)
    if file:
        uploads.append(file)
    if not uploads:
        raise HTTPException(status_code=400, detail="请选择要上传的知识库文档")

    added: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for upload in uploads:
        try:
            path = await save_upload_file(upload, "kb_upload")
            doc_id = kb.add_file_document(path, title=upload.filename or path.name, category=category or "上传文档")
            if not enabled:
                kb.set_document_enabled(doc_id, False)
            report = kb.get_document_parse_report(doc_id)
            added.append({"doc_id": doc_id, "file": upload.filename, "parse_report": report})
        except Exception as e:
            errors.append({"file": upload.filename or "", "error": str(e)})

    status_code = 207 if errors and added else (400 if errors and not added else 200)
    body = {
        "success": not errors,
        "added": added,
        "errors": errors,
        "doc_id": added[0]["doc_id"] if len(added) == 1 else None,
        "status": normalize_kb_status(kb.get_status()),
    }
    return JSONResponse(status_code=status_code, content=json_safe(body))


@app.delete("/api/kb/documents/{doc_id}")
@app.delete("/api/kb/document/{doc_id}")
def kb_delete_document(doc_id: int):
    kb = kb_service()
    kb.delete_document(doc_id)
    return JSONResponse(content=json_safe({"success": True, "status": normalize_kb_status(kb.get_status())}))


@app.post("/api/kb/documents/{doc_id}/toggle")
@app.post("/api/kb/document/{doc_id}/enabled")
def kb_toggle_document(
    doc_id: int,
    req: Optional[KBEnabledRequest] = Body(None),
    enabled: Optional[bool] = None,
):
    kb = kb_service()
    next_enabled = req.enabled if req is not None else (True if enabled is None else enabled)
    kb.set_document_enabled(doc_id, next_enabled)
    return JSONResponse(content=json_safe({"success": True, "enabled": next_enabled, "status": normalize_kb_status(kb.get_status())}))


@app.post("/api/kb/rebuild-folder")
def kb_rebuild_folder():
    from rag.generation_service import build_generation

    result = build_generation(publish=True)
    return JSONResponse(content=json_safe({"success": True, **result}))


@app.post("/api/kb/clear")
def kb_clear():
    kb = kb_service()
    kb.clear_kb()
    return JSONResponse(content=json_safe({"success": True, "status": normalize_kb_status(kb.get_status())}))

# =============================================================================
# Session APIs
# =============================================================================
def session_title(session: Dict[str, Any]) -> str:
    for message in session.get("messages", []):
        if message.get("role") == "user" and message.get("content"):
            text = str(message.get("content", "")).strip().replace("\n", " ")
            return text[:36] + ("..." if len(text) > 36 else "")
    return "新会话"


@app.get("/api/sessions")
def list_sessions():
    rows = []
    for sid, session in SESSIONS.items():
        messages = session.get("messages", []) or []
        rows.append({
            "session_id": sid,
            "title": session_title(session),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "message_count": len(messages),
            "has_last_result": session.get("last_result") is not None,
            "has_last_report": bool(session.get("last_report")),
        })
    rows.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return JSONResponse(content=json_safe({"sessions": rows}))


@app.post("/api/session/new")
def new_session():
    session_id = ensure_session()
    session = SESSIONS[session_id]
    return JSONResponse(content=json_safe({
        "session_id": session_id,
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
    }))


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    ensure_session(session_id)
    session = SESSIONS[session_id]
    return JSONResponse(content=json_safe({
        "session_id": session_id,
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "messages": session.get("messages", []),
        "last_result": session.get("last_result"),
        "last_report": session.get("last_report"),
    }))


@app.post("/api/session/{session_id}/messages")
def save_session_message(session_id: str, req: SessionMessageRequest):
    role = (req.role or "").strip() or "user"
    if role not in {"user", "assistant", "system"}:
        role = "user"
    content = (req.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content 不能为空")
    append_message(session_id, role, content)
    session = SESSIONS[session_id]
    return JSONResponse(content=json_safe({
        "ok": True,
        "session_id": session_id,
        "message_count": len(session.get("messages", [])),
        "updated_at": session.get("updated_at"),
    }))


@app.post("/api/session/clear")
def clear_session():
    SESSION_STATE["messages"] = []
    SESSION_STATE["last_result"] = None
    SESSION_STATE["last_report"] = None
    SESSION_STATE["batch_results"] = []
    SESSIONS.clear()
    save_sessions()
    return JSONResponse(content={"ok": True})
