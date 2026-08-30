from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


PARSER_NAME = "smart_pdf_router"
PARSER_VERSION = "1.0.0"
MIN_NATIVE_CHARS = 80
MIN_USABLE_CHARS = 12
MAX_GARBLED_RATIO = 0.02


class PdfParseError(RuntimeError):
    """Raised when a PDF cannot produce any trustworthy text."""


@dataclass
class OcrPageResult:
    text: str
    confidence: Optional[float] = None
    engine: str = "unknown"


class OcrProvider(Protocol):
    @property
    def available(self) -> bool: ...

    def parse_page(self, pdf_path: Path, page_number: int) -> OcrPageResult: ...


class UnavailableOcrProvider:
    """Safe default: report OCR demand instead of silently indexing empty text."""

    @property
    def available(self) -> bool:
        return False

    def parse_page(self, pdf_path: Path, page_number: int) -> OcrPageResult:
        raise RuntimeError("OCR provider is not configured")


@dataclass
class PdfPageResult:
    page_number: int
    parse_method: str
    status: str
    text: str
    text_char_count: int
    meaningful_char_count: int
    quality_score: float
    garbled_ratio: float
    image_count: int
    needs_ocr: bool
    ocr_confidence: Optional[float] = None
    warning: str = ""
    table_count: int = 0
    extracted_image_count: int = 0
    tables: List[Dict[str, Any]] = field(default_factory=list)
    assets: List[Dict[str, Any]] = field(default_factory=list)

    def public_dict(self, include_text: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_text:
            data.pop("text", None)
        return data


@dataclass
class PdfParseResult:
    source: str
    parser_name: str = PARSER_NAME
    parser_version: str = PARSER_VERSION
    status: str = "completed"
    pages: List[PdfPageResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_text: Optional[str] = field(default=None, repr=False)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def native_page_count(self) -> int:
        return sum(
            page.parse_method in {"native", "pymupdf_native"} for page in self.pages
        )

    @property
    def ocr_page_count(self) -> int:
        return sum(
            page.parse_method.startswith("ocr") or page.parse_method == "rapidocr"
            for page in self.pages
        )

    @property
    def needs_ocr_page_count(self) -> int:
        return sum(page.needs_ocr for page in self.pages)

    @property
    def warning_count(self) -> int:
        return len(self.warnings) + sum(bool(page.warning) for page in self.pages)

    @property
    def text(self) -> str:
        if self.document_text is not None:
            return self.document_text.strip()
        sections: List[str] = []
        for page in self.pages:
            value = (page.text or "").strip()
            if value:
                sections.append(f"[PDF_PAGE={page.page_number}]\n{value}")
        return "\n\n".join(sections)

    def public_dict(self, include_page_text: bool = False) -> Dict[str, Any]:
        return {
            "source": self.source,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "status": self.status,
            "page_count": self.page_count,
            "native_page_count": self.native_page_count,
            "ocr_page_count": self.ocr_page_count,
            "needs_ocr_page_count": self.needs_ocr_page_count,
            "warning_count": self.warning_count,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "pages": [page.public_dict(include_text=include_page_text) for page in self.pages],
        }


def _meaningful_char_count(text: str) -> int:
    return sum(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text or "")


def _garbled_ratio(text: str) -> float:
    value = text or ""
    if not value:
        return 0.0
    bad = value.count("\ufffd")
    bad += len(re.findall(r"\(cid:\d+\)", value, flags=re.IGNORECASE))
    bad += sum(ord(ch) < 32 and ch not in "\n\t\r" for ch in value)
    return round(bad / max(len(value), 1), 4)


def analyze_native_text(text: str) -> Dict[str, Any]:
    value = (text or "").strip()
    meaningful = _meaningful_char_count(value)
    garbled = _garbled_ratio(value)
    density_score = min(meaningful / MIN_NATIVE_CHARS, 1.0)
    quality_score = round(max(0.0, density_score * (1.0 - min(garbled * 10, 1.0))), 4)
    needs_ocr = meaningful < MIN_NATIVE_CHARS or garbled > MAX_GARBLED_RATIO

    if meaningful < MIN_USABLE_CHARS:
        reason = "页面几乎没有可用文字"
    elif garbled > MAX_GARBLED_RATIO:
        reason = f"乱码比例过高（{garbled:.1%}）"
    elif meaningful < MIN_NATIVE_CHARS:
        reason = f"有效文字偏少（{meaningful} 字符）"
    else:
        reason = ""

    return {
        "text_char_count": len(value),
        "meaningful_char_count": meaningful,
        "garbled_ratio": garbled,
        "quality_score": quality_score,
        "needs_ocr": needs_ocr,
        "reason": reason,
    }


def _page_image_count(page: Any) -> int:
    try:
        return len(list(page.images))
    except Exception:
        return 0


def parse_pdf(
    path: Path,
    ocr_provider: Optional[OcrProvider] = None,
) -> PdfParseResult:
    """Parse a PDF with the configured worker, retaining the legacy test hook."""
    if ocr_provider is None:
        try:
            from config import settings

            parser_mode = str(getattr(settings, "PDF_PARSER_MODE", "lightweight"))
        except Exception:
            parser_mode = "lightweight"
        if parser_mode == "lightweight":
            from rag.light_pdf_adapter import parse_with_lightweight_worker

            return parse_with_lightweight_worker(Path(path))

    from pypdf import PdfReader

    pdf_path = Path(path)
    provider = ocr_provider or UnavailableOcrProvider()
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PdfParseError(f"PDF 无法打开：{exc}") from exc

    result = PdfParseResult(source=str(pdf_path))
    if not reader.pages:
        raise PdfParseError("PDF 没有页面")

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            native_text = page.extract_text() or ""
        except Exception as exc:
            native_text = ""
            result.warnings.append(f"第 {page_number} 页原生文本提取异常：{exc}")

        native = analyze_native_text(native_text)
        image_count = _page_image_count(page)
        if not native["needs_ocr"]:
            result.pages.append(PdfPageResult(
                page_number=page_number,
                parse_method="native",
                status="completed",
                text=native_text.strip(),
                text_char_count=native["text_char_count"],
                meaningful_char_count=native["meaningful_char_count"],
                quality_score=native["quality_score"],
                garbled_ratio=native["garbled_ratio"],
                image_count=image_count,
                needs_ocr=False,
            ))
            continue

        if provider.available:
            try:
                ocr = provider.parse_page(pdf_path, page_number)
                ocr_metrics = analyze_native_text(ocr.text)
                usable = ocr_metrics["meaningful_char_count"] >= MIN_USABLE_CHARS
                warning = "" if usable else "OCR 后仍没有足够的可用文字"
                result.pages.append(PdfPageResult(
                    page_number=page_number,
                    parse_method=f"ocr:{ocr.engine}",
                    status="completed" if usable else "needs_review",
                    text=(ocr.text or "").strip() if usable else native_text.strip(),
                    text_char_count=ocr_metrics["text_char_count"],
                    meaningful_char_count=ocr_metrics["meaningful_char_count"],
                    quality_score=ocr_metrics["quality_score"],
                    garbled_ratio=ocr_metrics["garbled_ratio"],
                    image_count=image_count,
                    needs_ocr=not usable,
                    ocr_confidence=ocr.confidence,
                    warning=warning,
                ))
                continue
            except Exception as exc:
                result.warnings.append(f"第 {page_number} 页 OCR 执行失败：{exc}")

        keep_native = native["meaningful_char_count"] >= MIN_USABLE_CHARS
        result.pages.append(PdfPageResult(
            page_number=page_number,
            parse_method="native_low_quality" if keep_native else "none",
            status="needs_ocr",
            text=native_text.strip() if keep_native else "",
            text_char_count=native["text_char_count"],
            meaningful_char_count=native["meaningful_char_count"],
            quality_score=native["quality_score"],
            garbled_ratio=native["garbled_ratio"],
            image_count=image_count,
            needs_ocr=True,
            warning=f"{native['reason']}；OCR 引擎尚未配置",
        ))

    if not result.text.strip():
        result.status = "failed"
        raise PdfParseError("PDF 没有提取到可入库文字；疑似扫描件，请配置 OCR 后重试")
    if result.needs_ocr_page_count:
        result.status = "needs_ocr"
    elif result.warning_count:
        result.status = "completed_with_warnings"
    else:
        result.status = "completed"
    return result
