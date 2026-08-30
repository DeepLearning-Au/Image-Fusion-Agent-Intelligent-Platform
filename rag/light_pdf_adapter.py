from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from rag.pdf_parser import PdfPageResult, PdfParseError, PdfParseResult


PROJECT_DIR = Path(__file__).resolve().parents[1]
ADAPTER_VERSION = "1.0.0"


def _setting(name: str, default: Any) -> Any:
    try:
        from config import settings

        return getattr(settings, name, default)
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser_runtime_status() -> Dict[str, Any]:
    python_path = Path(
        _setting(
            "PDF_PARSER_WORKER_PYTHON",
            PROJECT_DIR / ".venv-pdf-parser" / "Scripts" / "python.exe",
        )
    )
    worker_path = Path(
        _setting(
            "PDF_PARSER_WORKER_SCRIPT",
            PROJECT_DIR / "workers" / "light_pdf_worker.py",
        )
    )
    return {
        "mode": str(_setting("PDF_PARSER_MODE", "lightweight")),
        "parser_name": "pymupdf_rapidocr_pdfplumber",
        "adapter_version": ADAPTER_VERSION,
        "python_path": str(python_path),
        "worker_path": str(worker_path),
        "available": python_path.exists() and worker_path.exists(),
        "external_service": False,
    }


def _read_cache(result_path: Path, source_hash: str) -> Dict[str, Any] | None:
    if not result_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        not payload.get("success")
        or payload.get("source_sha256") != source_hash
        or payload.get("parser_version") != ADAPTER_VERSION
        or not str(payload.get("document_text", "")).strip()
    ):
        return None
    return {**payload, "cache_hit": True}


def _run_worker(path: Path, source_hash: str, force_ocr: bool = False) -> Dict[str, Any]:
    runtime = parser_runtime_status()
    if not runtime["available"]:
        raise PdfParseError(
            "轻量 PDF Worker 未安装，请运行 scripts/setup_pdf_parser_worker.ps1"
        )
    cache_root = Path(
        _setting("PDF_PARSER_CACHE_DIR", PROJECT_DIR / "storage" / "pdf_parser_cache")
    )
    cache_key = f"{source_hash[:20]}-v{ADAPTER_VERSION}-ocr{int(force_ocr)}"
    output_dir = cache_root / cache_key
    result_path = output_dir / "result.json"
    cached = _read_cache(result_path, source_hash)
    if cached:
        return cached

    assets_root = Path(
        _setting(
            "PDF_PARSER_ASSETS_DIR",
            PROJECT_DIR / "data" / "knowledge_assets" / "documents",
        )
    )
    assets_dir = assets_root / source_hash[:20]
    asset_url_prefix = f"/knowledge-assets/documents/{source_hash[:20]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        runtime["python_path"],
        runtime["worker_path"],
        "--input", str(path.resolve()),
        "--result", str(result_path.resolve()),
        "--assets-dir", str(assets_dir.resolve()),
        "--asset-url-prefix", asset_url_prefix,
        "--render-scale", str(float(_setting("PDF_OCR_RENDER_SCALE", 2.0))),
    ]
    if force_ocr:
        command.append("--force-ocr")
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    timeout = int(_setting("PDF_PARSER_TIMEOUT_SECONDS", 180))
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PdfParseError(
            f"轻量 PDF Worker 没有返回有效结果：{completed.stderr.strip() or exc}"
        ) from exc
    if completed.returncode != 0 or not payload.get("success"):
        raise PdfParseError(
            payload.get("error") or completed.stderr.strip() or "轻量 PDF 解析失败"
        )
    return {**payload, "cache_hit": False}


def parse_with_lightweight_worker(path: Path, force_ocr: bool = False) -> PdfParseResult:
    pdf_path = Path(path)
    source_hash = sha256_file(pdf_path)
    payload = _run_worker(pdf_path, source_hash=source_hash, force_ocr=force_ocr)
    pages = [
        PdfPageResult(
            page_number=int(page["page_number"]),
            parse_method=str(page["parse_method"]),
            status=str(page["status"]),
            text=str(page.get("text", "")),
            text_char_count=int(page.get("text_char_count", 0)),
            meaningful_char_count=int(page.get("meaningful_char_count", 0)),
            quality_score=float(page.get("quality_score", 0)),
            garbled_ratio=float(page.get("garbled_ratio", 0)),
            image_count=int(page.get("image_count", 0)),
            needs_ocr=bool(page.get("needs_ocr", False)),
            ocr_confidence=page.get("ocr_confidence"),
            warning=str(page.get("warning", "")),
            table_count=int(page.get("table_count", 0)),
            extracted_image_count=int(page.get("extracted_image_count", 0)),
            tables=list(page.get("tables", [])),
            assets=list(page.get("assets", [])),
        )
        for page in payload.get("pages", [])
    ]
    result = PdfParseResult(
        source=str(pdf_path),
        parser_name=str(payload.get("parser_name", "pymupdf_rapidocr_pdfplumber")),
        parser_version=str(payload.get("parser_version", ADAPTER_VERSION)),
        status=str(payload.get("status", "completed")),
        pages=pages,
        document_text=str(payload.get("document_text", "")),
        metadata={
            "cache_hit": bool(payload.get("cache_hit")),
            "external_service": False,
            "table_count": int(payload.get("table_count", 0)),
            "extracted_image_count": int(payload.get("extracted_image_count", 0)),
            "dependencies": dict(payload.get("dependencies", {})),
        },
    )
    if not result.text.strip():
        raise PdfParseError("轻量 PDF Worker 没有生成可入库文本")
    return result
