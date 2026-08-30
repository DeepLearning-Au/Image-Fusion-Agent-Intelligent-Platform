from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


WORKER_NAME = "pymupdf_rapidocr_pdfplumber"
WORKER_VERSION = "1.0.0"
MIN_NATIVE_CHARS = 80
MIN_USABLE_CHARS = 12
MAX_GARBLED_RATIO = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local lightweight PDF parsing worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--asset-url-prefix", required=True)
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--render-scale", type=float, default=2.0)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def meaningful_char_count(text: str) -> int:
    return sum(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text or "")


def analyze_text(text: str) -> Dict[str, Any]:
    value = (text or "").strip()
    meaningful = meaningful_char_count(value)
    bad = value.count("\ufffd") + len(re.findall(r"\(cid:\d+\)", value, re.IGNORECASE))
    garbled_ratio = round(bad / max(len(value), 1), 4)
    density = min(meaningful / MIN_NATIVE_CHARS, 1.0)
    quality = round(max(0.0, density * (1.0 - min(garbled_ratio * 10, 1.0))), 4)
    return {
        "text_char_count": len(value),
        "meaningful_char_count": meaningful,
        "garbled_ratio": garbled_ratio,
        "quality_score": quality,
        "needs_ocr": meaningful < MIN_NATIVE_CHARS or garbled_ratio > MAX_GARBLED_RATIO,
    }


def clean_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", "\\|")


def normalize_table(rows: Sequence[Sequence[Any]]) -> List[List[str]]:
    normalized = [[clean_cell(cell) for cell in row] for row in rows if row]
    normalized = [row for row in normalized if any(row)]
    if not normalized:
        return []
    width = max(len(row) for row in normalized)
    return [row + [""] * (width - len(row)) for row in normalized]


def table_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    header = list(rows[0])
    body = list(rows[1:])
    if len(rows) == 1:
        body = [[]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body if row)
    return "\n".join(lines)


def extract_tables(plumber_page: Any) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    for index, raw in enumerate(plumber_page.extract_tables() or [], start=1):
        rows = normalize_table(raw or [])
        if len(rows) < 2 or max(len(row) for row in rows) < 2:
            continue
        tables.append({"index": index, "rows": rows, "markdown": table_to_markdown(rows)})
    return tables


def extract_images(
    document: Any,
    page: Any,
    page_number: int,
    assets_dir: Path,
    asset_url_prefix: str,
    seen_hashes: set[str],
) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for image_number, image_info in enumerate(page.get_images(full=True), start=1):
        xref = int(image_info[0])
        try:
            extracted = document.extract_image(xref)
            data = extracted["image"]
            width = int(extracted.get("width", 0))
            height = int(extracted.get("height", 0))
            extension = str(extracted.get("ext") or "png").lower()
        except Exception:
            continue
        if width < 100 or height < 80 or width * height < 20000:
            continue
        try:
            from PIL import Image, ImageStat

            preview = Image.open(io.BytesIO(data)).convert("L").resize((64, 64))
            statistics = ImageStat.Stat(preview)
            if statistics.mean[0] > 248 and statistics.stddev[0] < 10:
                continue
        except Exception:
            pass
        digest = sha256_bytes(data)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        name = f"page_{page_number:03d}_image_{image_number:02d}_{digest[:8]}.{extension}"
        target = assets_dir / name
        target.write_bytes(data)
        assets.append({
            "media_type": "document_image",
            "page_number": page_number,
            "width": width,
            "height": height,
            "sha256": digest,
            "path": str(target),
            "uri": f"{asset_url_prefix.rstrip('/')}/{name}",
        })
    return assets


def ocr_page(page: Any, engine: Any, render_scale: float) -> tuple[str, float | None]:
    import pymupdf

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(render_scale, render_scale),
        alpha=False,
    )
    output = engine(pixmap.tobytes("png"))
    if output is None or output.txts is None or len(output.txts) == 0:
        return "", None
    items = []
    for box, text, score in zip(output.boxes, output.txts, output.scores):
        x = min(float(point[0]) for point in box)
        y = min(float(point[1]) for point in box)
        items.append((round(y / 16) * 16, x, str(text).strip(), float(score)))
    items.sort(key=lambda item: (item[0], item[1]))
    text = "\n".join(item[2] for item in items if item[2])
    scores = [item[3] for item in items if item[2]]
    confidence = round(sum(scores) / len(scores), 4) if scores else None
    return text, confidence


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_document_text(pages: Iterable[Dict[str, Any]]) -> str:
    sections: List[str] = []
    for page in pages:
        parts = [f"[PDF_PAGE={page['page_number']}]", page["text"].strip()]
        for table in page.get("tables", []):
            parts.extend([f"\n### 第 {page['page_number']} 页表格 {table['index']}", table["markdown"]])
        for asset in page.get("assets", []):
            parts.append(
                f"\n[DOCUMENT_IMAGE page={page['page_number']} uri={asset['uri']} "
                f"size={asset['width']}x{asset['height']}]"
            )
        sections.append("\n".join(part for part in parts if part))
    return "\n\n".join(sections).strip()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    result_path = Path(args.result).resolve()
    assets_dir = Path(args.assets_dir).resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(input_path)
    base = {
        "source": str(input_path),
        "source_sha256": source_hash,
        "parser_name": WORKER_NAME,
        "parser_version": WORKER_VERSION,
    }
    try:
        import pdfplumber
        import pymupdf

        document = pymupdf.open(input_path)
        plumber_document = pdfplumber.open(input_path)
        if document.page_count != len(plumber_document.pages):
            raise RuntimeError("PyMuPDF 与 pdfplumber 返回的页数不一致")

        pages: List[Dict[str, Any]] = []
        seen_hashes: set[str] = set()
        ocr_engine = None
        for page_index in range(document.page_count):
            page_number = page_index + 1
            page = document[page_index]
            native_text = page.get_text("text", sort=True) or ""
            native_metrics = analyze_text(native_text)
            use_ocr = bool(args.force_ocr or native_metrics["needs_ocr"])
            ocr_confidence = None
            warning = ""
            if use_ocr:
                if ocr_engine is None:
                    from rapidocr import RapidOCR

                    ocr_engine = RapidOCR()
                final_text, ocr_confidence = ocr_page(page, ocr_engine, args.render_scale)
                final_metrics = analyze_text(final_text)
                parse_method = "rapidocr"
                if final_metrics["meaningful_char_count"] < MIN_USABLE_CHARS:
                    warning = "RapidOCR 后仍没有足够的可用文字"
            else:
                final_text = native_text.strip()
                final_metrics = native_metrics
                parse_method = "pymupdf_native"

            tables = extract_tables(plumber_document.pages[page_index])
            assets = extract_images(
                document,
                page,
                page_number,
                assets_dir,
                args.asset_url_prefix,
                seen_hashes,
            )
            pages.append({
                "page_number": page_number,
                "parse_method": parse_method,
                "status": "needs_review" if warning else "completed",
                "text": final_text,
                **final_metrics,
                "needs_ocr": bool(warning),
                "ocr_confidence": ocr_confidence,
                "image_count": len(page.get_images(full=True)),
                "extracted_image_count": len(assets),
                "table_count": len(tables),
                "tables": tables,
                "assets": assets,
                "warning": warning,
            })

        document_text = build_document_text(pages)
        if meaningful_char_count(document_text) < MIN_USABLE_CHARS:
            raise RuntimeError("解析结果没有足够的可用文字")
        payload = {
            **base,
            "success": True,
            "status": "completed_with_warnings" if any(page["warning"] for page in pages) else "completed",
            "page_count": len(pages),
            "native_page_count": sum(page["parse_method"] == "pymupdf_native" for page in pages),
            "ocr_page_count": sum(page["parse_method"] == "rapidocr" for page in pages),
            "needs_ocr_page_count": sum(page["needs_ocr"] for page in pages),
            "warning_count": sum(bool(page["warning"]) for page in pages),
            "table_count": sum(page["table_count"] for page in pages),
            "extracted_image_count": sum(page["extracted_image_count"] for page in pages),
            "document_text": document_text,
            "pages": pages,
            "dependencies": {
                "pymupdf": version("pymupdf"),
                "pdfplumber": version("pdfplumber"),
                "rapidocr": version("rapidocr"),
                "onnxruntime": version("onnxruntime"),
            },
        }
        write_json(result_path, payload)
        plumber_document.close()
        document.close()
        return 0
    except Exception as exc:
        write_json(result_path, {**base, "success": False, "status": "failed", "error": str(exc)})
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
