from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from config import settings


class BGEReranker:
    """使用 Transformers 直接运行 BGE Cross-Encoder，避免额外引入重型框架。"""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
        local_files_only: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        if self.device.startswith("cuda"):
            self.model.half()
        self.model.eval()

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        import torch

        scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            pairs = [[query, text] for text in batch_texts]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits.reshape(-1).float()
                batch_scores = torch.sigmoid(logits).cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
        return scores


_LOCK = threading.Lock()
_INSTANCE: Optional[BGEReranker] = None
_INSTANCE_KEY: Optional[tuple] = None


def get_reranker(*, allow_download: bool = False) -> BGEReranker:
    global _INSTANCE, _INSTANCE_KEY
    key = (
        settings.RERANKER_MODEL,
        settings.RERANKER_DEVICE,
        settings.RERANKER_BATCH_SIZE,
        settings.RERANKER_MAX_LENGTH,
    )
    with _LOCK:
        if _INSTANCE is None or _INSTANCE_KEY != key:
            _INSTANCE = BGEReranker(
                settings.RERANKER_MODEL,
                device=settings.RERANKER_DEVICE,
                batch_size=settings.RERANKER_BATCH_SIZE,
                max_length=settings.RERANKER_MAX_LENGTH,
                local_files_only=(settings.RERANKER_LOCAL_FILES_ONLY and not allow_download),
            )
            _INSTANCE_KEY = key
    return _INSTANCE


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    reranker: Any = None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    engine = reranker or get_reranker()
    texts = [str(item.get("text") or item.get("content") or "") for item in candidates]
    scores = engine.score_pairs(query, texts)
    if len(scores) != len(candidates):
        raise RuntimeError("BGE Reranker 返回分数数量不正确")

    ranked = []
    for item, score in zip(candidates, scores):
        value = dict(item)
        value["reranker_score"] = round(float(score), 6)
        value["final_score"] = value["reranker_score"]
        ranked.append(value)
    ranked.sort(key=lambda item: item["reranker_score"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["reranker_rank"] = rank
    return ranked[:top_k]


def reranker_status() -> Dict[str, Any]:
    cached = False
    cache_path = ""
    error = ""
    try:
        from huggingface_hub import snapshot_download

        cache_path = snapshot_download(
            repo_id=settings.RERANKER_MODEL,
            local_files_only=True,
        )
        cached = True
    except Exception as exc:
        error = str(exc)
    return {
        "enabled": settings.RERANKER_ENABLED,
        "model": settings.RERANKER_MODEL,
        "device": settings.RERANKER_DEVICE,
        "cached": cached,
        "loaded": _INSTANCE is not None,
        "cache_path": cache_path,
        "message": "模型尚未下载" if not cached else "模型已就绪",
        "error": error if not cached else "",
    }


def preload_reranker(*, allow_download: bool = False) -> Dict[str, Any]:
    get_reranker(allow_download=allow_download)
    return reranker_status()
