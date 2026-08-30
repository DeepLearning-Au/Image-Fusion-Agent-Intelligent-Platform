import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag.bge_reranker import get_reranker, preload_reranker


if __name__ == "__main__":
    result = preload_reranker(allow_download=True)
    result["smoke_test_scores"] = get_reranker().score_pairs(
        "夜间发现目标应该使用什么设备",
        ["红外设备适合夜间发现热目标", "可见光相机用于白天记录颜色纹理"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
