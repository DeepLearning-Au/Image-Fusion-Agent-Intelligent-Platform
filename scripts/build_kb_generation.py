import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rag import sqlite_kb
from rag.generation_service import build_generation, list_generations


if __name__ == "__main__":
    result = build_generation(publish=True)
    print(json.dumps({
        "build": result,
        "active_status": sqlite_kb.get_status(),
        "generations": list_generations(),
    }, ensure_ascii=False, indent=2))
