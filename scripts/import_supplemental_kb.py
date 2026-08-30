import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import settings
from rag import sqlite_kb
from rag.enterprise_retrieval import enterprise_search
        from rag.vector_store import sync_chunk_embeddings


if __name__ == "__main__":
    reports = [
        sqlite_kb.import_folder_incremental_report(folder)
        for folder in settings.SUPPLEMENTAL_KNOWLEDGE_DIRS
    ]
    output = {"reports": reports}
    try:
        output["vector_index"] = sync_chunk_embeddings()
    except Exception as exc:
        output["vector_error"] = str(exc)
    output["knowledge_base"] = sqlite_kb.get_status()
    try:
        smoke_result = enterprise_search(
            "夜间低照度环境应该选择什么成像设备，设备应如何维护？",
            top_k=3,
        )
        output["retrieval_smoke_test"] = {
            "retrieval": smoke_result.get("retrieval", {}),
            "sources": [
                {
                    "doc_title": item.get("doc_title"),
                    "final_score": item.get("final_score"),
                    "rrf_score": item.get("rrf_score"),
                }
                for item in smoke_result.get("sources", [])
            ],
        }
    except Exception as exc:
        output["retrieval_smoke_error"] = str(exc)
    print(json.dumps(output, ensure_ascii=False, indent=2))
