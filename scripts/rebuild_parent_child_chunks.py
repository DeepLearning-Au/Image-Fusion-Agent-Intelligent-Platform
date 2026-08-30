import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag import sqlite_kb
from rag.enterprise_retrieval import enterprise_search
        from rag.vector_store import sync_chunk_embeddings


if __name__ == "__main__":
    output = {"rechunk": sqlite_kb.rechunk_all_documents()}
    try:
        output["vector_index"] = sync_chunk_embeddings()
        smoke = enterprise_search(
            "推荐适合夜间低照度检测并支持后续图像融合的设备",
            top_k=3,
        )
        output["retrieval"] = {
            "pipeline": smoke.get("retrieval", {}),
            "sources": [
                {
                    "doc_title": item.get("doc_title"),
                    "title_path": item.get("title_path"),
                    "matched_child_id": item.get("matched_child_id"),
                    "parent_id": item.get("parent_id"),
                    "parent_chars": len(item.get("text") or ""),
                }
                for item in smoke.get("sources", [])
            ],
        }
    except Exception as exc:
        output["pipeline_error"] = str(exc)
    output["knowledge_base"] = sqlite_kb.get_status()
    print(json.dumps(output, ensure_ascii=False, indent=2))
