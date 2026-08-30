from pathlib import Path
import os
from urllib.parse import quote
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"


def _load_yaml(file_name: str) -> dict:
    path = CONFIG_DIR / file_name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _load_env_file(path: Path) -> None:
    """加载本地部署密钥，但不覆盖系统已经注入的环境变量。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _path(value) -> Path:
    value = str(value).replace("{ROOT_DIR}", str(ROOT_DIR))
    p = Path(value)
    return p if p.is_absolute() else ROOT_DIR / p


def _get(d: dict, keys: list, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


_agent_conf = _load_yaml("agent.yaml")
_llm_conf = _load_yaml("llm.yaml")
_rag_conf = _load_yaml("rag.yaml")
_prompt_conf = _load_yaml("prompt.yaml")
_fusion_conf = _load_yaml("fusion.yaml")
_queue_conf = _load_yaml("queue.yaml")
_load_env_file(ROOT_DIR / "deploy" / "redis" / ".env")


# =========================================================
# Redis / Celery任务队列配置
# =========================================================

TASK_QUEUE_ENABLED = os.getenv("TASK_QUEUE_ENABLED", "1") == "1"
REDIS_HOST = os.getenv("REDIS_HOST", _get(_queue_conf, ["redis", "host"], "127.0.0.1"))
REDIS_PORT = int(os.getenv("REDIS_PORT", _get(_queue_conf, ["redis", "port"], 6380)))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
CELERY_BROKER_DB = int(
    os.getenv("CELERY_BROKER_DB", _get(_queue_conf, ["redis", "broker_db"], 1))
)
CELERY_RESULT_DB = int(
    os.getenv("CELERY_RESULT_DB", _get(_queue_conf, ["redis", "result_db"], 2))
)


def _redis_url(database: int) -> str:
    auth = f":{quote(REDIS_PASSWORD, safe='')}@" if REDIS_PASSWORD else ""
    return f"redis://{auth}{REDIS_HOST}:{REDIS_PORT}/{database}"


CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", _redis_url(CELERY_BROKER_DB))
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", _redis_url(CELERY_RESULT_DB))
REDIS_HEALTH_URL = _redis_url(0)
REDIS_SOCKET_TIMEOUT_SECONDS = int(
    _get(_queue_conf, ["redis", "socket_timeout_seconds"], 5)
)
CELERY_RESULT_EXPIRES_SECONDS = int(
    _get(_queue_conf, ["celery", "result_expires_seconds"], 86400)
)
CELERY_VISIBILITY_TIMEOUT_SECONDS = int(
    _get(_queue_conf, ["celery", "visibility_timeout_seconds"], 3600)
)
CELERY_SOFT_TIME_LIMIT_SECONDS = int(
    _get(_queue_conf, ["celery", "soft_time_limit_seconds"], 1500)
)
CELERY_HARD_TIME_LIMIT_SECONDS = int(
    _get(_queue_conf, ["celery", "hard_time_limit_seconds"], 1800)
)
CELERY_MAX_RETRIES = int(_get(_queue_conf, ["celery", "max_retries"], 2))
CELERY_RETRY_BACKOFF_MAX_SECONDS = int(
    _get(_queue_conf, ["celery", "retry_backoff_max_seconds"], 60)
)
CELERY_WORKER_PREFETCH_MULTIPLIER = int(
    _get(_queue_conf, ["celery", "worker_prefetch_multiplier"], 1)
)
CELERY_FUSION_QUEUE = str(_get(_queue_conf, ["routes", "fusion_queue"], "gpu"))
CELERY_MAINTENANCE_QUEUE = str(
    _get(_queue_conf, ["routes", "maintenance_queue"], "maintenance")
)


# =========================================================
# Agent / 业务配置
# =========================================================

AGENT_USER_CITY = os.getenv(
    "AGENT_USER_CITY",
    _get(_agent_conf, ["user", "default_city"], "烟台"),
)
AGENT_USER_ID = os.getenv(
    "AGENT_USER_ID",
    _get(_agent_conf, ["user", "default_user_id"], "customer_001"),
)
EXTERNAL_DATA_PATH = _path(
    _get(_agent_conf, ["external", "data_path"], "data/external/records.csv")
)


# =========================================================
# 大模型 / Embedding 配置
# =========================================================

_api_key_env_name = _get(_llm_conf, ["dashscope", "api_key_env"], "DASHSCOPE_API_KEY")
DASHSCOPE_API_KEY = os.getenv(_api_key_env_name, "")

DASHSCOPE_BASE_URL = _get(
    _llm_conf,
    ["dashscope", "base_url"],
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MODEL = os.getenv("LLM_MODEL", _get(_llm_conf, ["llm", "model"], "qwen3-max"))
LLM_TEMPERATURE = float(_get(_llm_conf, ["llm", "temperature"], 0.2))
LLM_TIMEOUT_SECONDS = int(_get(_llm_conf, ["llm", "timeout_seconds"], 90))
LLM_MAX_RETRIES = int(_get(_llm_conf, ["llm", "max_retries"], 2))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    _get(_llm_conf, ["embedding", "model"], "text-embedding-v4"),
)
EMBEDDING_BATCH_SIZE = int(_get(_llm_conf, ["embedding", "batch_size"], 10))


# =========================================================
# RAG / 向量库配置
# =========================================================

KNOWLEDGE_DIR = _path(_get(_rag_conf, ["paths", "knowledge_dir"], "data/knowledge_docs"))
SUPPLEMENTAL_KNOWLEDGE_DIRS = [
    _path(value) for value in _get(
        _rag_conf,
        ["paths", "supplemental_knowledge_dirs"],
        [],
    )
]
RAG_INDEX_DIR = _path(_get(_rag_conf, ["paths", "rag_index_dir"], "storage/rag_index"))
CHROMA_DIR = _path(_get(_rag_conf, ["paths", "chroma_dir"], "storage/chroma"))
KNOWLEDGE_MANIFEST_PATH = _path(
    _get(_rag_conf, ["paths", "manifest_path"], "storage/knowledge_manifest.json")
)

VECTOR_STORE_BACKEND = _get(_rag_conf, ["vector_store", "backend"], "pickle")
MILVUS_URI = os.getenv("MILVUS_URI", _get(_rag_conf, ["milvus", "uri"], "http://127.0.0.1:19530"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION = os.getenv(
    "MILVUS_COLLECTION", _get(_rag_conf, ["milvus", "collection"], "mambadfuse_chunks")
)
MILVUS_METRIC_TYPE = str(_get(_rag_conf, ["milvus", "metric_type"], "COSINE")).upper()
MILVUS_INDEX_TYPE = os.getenv(
    "MILVUS_INDEX_TYPE", str(_get(_rag_conf, ["milvus", "index_type"], "HNSW"))
).upper()
MILVUS_TIMEOUT_SECONDS = int(_get(_rag_conf, ["milvus", "timeout_seconds"], 30))
MILVUS_UPSERT_BATCH_SIZE = int(_get(_rag_conf, ["milvus", "upsert_batch_size"], 128))
MILVUS_HNSW_M = int(_get(_rag_conf, ["milvus", "hnsw", "M"], 32))
MILVUS_HNSW_EF_CONSTRUCTION = int(
    _get(_rag_conf, ["milvus", "hnsw", "ef_construction"], 200)
)
MILVUS_HNSW_EF_SEARCH = int(_get(_rag_conf, ["milvus", "hnsw", "ef_search"], 128))
MILVUS_IVF_NLIST = int(_get(_rag_conf, ["milvus", "ivf_flat", "nlist"], 64))
MILVUS_IVF_NPROBE = int(_get(_rag_conf, ["milvus", "ivf_flat", "nprobe"], 16))
VECTOR_INDEX_FILE = _get(_rag_conf, ["vector_store", "index_file"], "optisar_rag_index.pkl")
RAG_TOP_K = int(_get(_rag_conf, ["vector_store", "top_k"], 5))
RAG_CANDIDATE_K = int(_get(_rag_conf, ["vector_store", "candidate_k"], 10))
RAG_MIN_RELEVANCE_SCORE = float(_get(_rag_conf, ["vector_store", "min_relevance_score"], 0.0))
RAG_RRF_K = int(_get(_rag_conf, ["hybrid_retrieval", "rrf_k"], 60))
RAG_EVALUATION_ENABLED = bool(_get(_rag_conf, ["evaluation", "enabled"], True))
RAG_EVAL_SET_PATH = _path(
    _get(_rag_conf, ["evaluation", "eval_set"], "storage/rag_eval_set.jsonl")
)
RAG_EVAL_TOP_K = int(_get(_rag_conf, ["evaluation", "top_k"], 5))
RAG_EVAL_MIN_QUESTIONS = int(_get(_rag_conf, ["evaluation", "min_questions"], 30))
RAG_EVAL_THRESHOLDS = {
    "recall_at_k": float(_get(_rag_conf, ["evaluation", "thresholds", "recall_at_k"], 0.80)),
    "mrr": float(_get(_rag_conf, ["evaluation", "thresholds", "mrr"], 0.65)),
    "ndcg_at_k": float(_get(_rag_conf, ["evaluation", "thresholds", "ndcg_at_k"], 0.70)),
}
RAG_ANSWER_EVALUATION_ENABLED = bool(
    _get(_rag_conf, ["evaluation", "answer_quality", "enabled"], True)
)
RAG_ANSWER_EVAL_SET_PATH = _path(
    _get(
        _rag_conf,
        ["evaluation", "answer_quality", "eval_set"],
        "storage/rag_answer_eval_set.jsonl",
    )
)
RAG_ANSWER_EVAL_MIN_QUESTIONS = int(
    _get(_rag_conf, ["evaluation", "answer_quality", "min_questions"], 12)
)
RAG_ANSWER_EVAL_THRESHOLDS = {
    "answer_relevance": float(_get(_rag_conf, ["evaluation", "answer_quality", "thresholds", "answer_relevance"], 0.80)),
    "faithfulness": float(_get(_rag_conf, ["evaluation", "answer_quality", "thresholds", "faithfulness"], 0.85)),
    "citation_completeness": float(_get(_rag_conf, ["evaluation", "answer_quality", "thresholds", "citation_completeness"], 0.80)),
    "refusal_correctness": float(_get(_rag_conf, ["evaluation", "answer_quality", "thresholds", "refusal_correctness"], 0.90)),
}
RERANKER_ENABLED = bool(_get(_rag_conf, ["reranker", "enabled"], True))
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    _get(_rag_conf, ["reranker", "model"], "BAAI/bge-reranker-base"),
)
RERANKER_CANDIDATE_K = int(_get(_rag_conf, ["reranker", "candidate_k"], 20))
RERANKER_BATCH_SIZE = int(_get(_rag_conf, ["reranker", "batch_size"], 8))
RERANKER_MAX_LENGTH = int(_get(_rag_conf, ["reranker", "max_length"], 512))
RERANKER_DEVICE = os.getenv(
    "RERANKER_DEVICE",
    _get(_rag_conf, ["reranker", "device"], "auto"),
)
RERANKER_LOCAL_FILES_ONLY = bool(
    _get(_rag_conf, ["reranker", "local_files_only"], True)
)

ALLOW_KNOWLEDGE_FILE_TYPE = _get(
    _rag_conf,
    ["loader", "allow_file_types"],
    ["txt", "md", "pdf", "csv", "docx", "py", "yaml", "yml"],
)

PDF_PARSER_MODE = os.getenv(
    "PDF_PARSER_MODE",
    _get(_rag_conf, ["pdf_parser", "mode"], "lightweight"),
)
PDF_PARSER_WORKER_PYTHON = _path(
    _get(_rag_conf, ["pdf_parser", "worker_python"], ".venv-pdf-parser/Scripts/python.exe")
)
PDF_PARSER_WORKER_SCRIPT = _path(
    _get(_rag_conf, ["pdf_parser", "worker_script"], "workers/light_pdf_worker.py")
)
PDF_PARSER_CACHE_DIR = _path(
    _get(_rag_conf, ["pdf_parser", "cache_dir"], "storage/pdf_parser_cache")
)
PDF_PARSER_ASSETS_DIR = _path(
    _get(_rag_conf, ["pdf_parser", "assets_dir"], "data/knowledge_assets/documents")
)
PDF_PARSER_TIMEOUT_SECONDS = int(
    _get(_rag_conf, ["pdf_parser", "timeout_seconds"], 180)
)
PDF_OCR_RENDER_SCALE = float(
    _get(_rag_conf, ["pdf_parser", "ocr_render_scale"], 2.0)
)

PARENT_CHUNK_SIZE = int(_get(_rag_conf, ["splitter", "parent_chunk_size"], 2000))
CHILD_CHUNK_SIZE = int(_get(_rag_conf, ["splitter", "child_chunk_size"], 500))
CHILD_CHUNK_OVERLAP = int(_get(_rag_conf, ["splitter", "child_chunk_overlap"], 80))
CHUNKING_VERSION = _get(_rag_conf, ["splitter", "chunking_version"], "parent_child_v1")
# 兼容旧代码。
CHUNK_SIZE = CHILD_CHUNK_SIZE
CHUNK_OVERLAP = CHILD_CHUNK_OVERLAP
SEPARATORS = _get(_rag_conf, ["splitter", "separators"], ["\n\n", "\n", "。", "；", "，", " ", ""])


# =========================================================
# Prompt 文件配置
# =========================================================

PROMPT_DIR = _path(_get(_prompt_conf, ["paths", "prompt_dir"], "prompts"))
MAIN_PROMPT_PATH = _path(_get(_prompt_conf, ["paths", "main_prompt"], "prompts/main_prompt.txt"))
RAG_SUMMARIZE_PROMPT_PATH = _path(
    _get(_prompt_conf, ["paths", "rag_summarize_prompt"], "prompts/rag_summarize.txt")
)
REPORT_PROMPT_PATH = _path(_get(_prompt_conf, ["paths", "report_prompt"], "prompts/report_prompt.txt"))


# =========================================================
# 融合模型配置
# =========================================================

FUSION_BACKEND = _get(_fusion_conf, ["fusion", "backend"], "MambaDFuse")

_USE_CPU = os.getenv("USE_CPU", "0") == "1"
_device_setting = _get(_fusion_conf, ["fusion", "device"], "auto")
DEVICE = "cpu" if _USE_CPU or _device_setting == "cpu" else "cuda"

TILE_SIZE = int(_get(_fusion_conf, ["fusion", "tile_size"], 256))
OVERLAP = int(_get(_fusion_conf, ["fusion", "overlap"], 32))
USE_TILE = bool(_get(_fusion_conf, ["fusion", "use_tile"], False))


def _detect_mambadfuse_root() -> Path:
    candidates = _get(
        _fusion_conf,
        ["mambadfuse", "root_candidates"],
        ["third_party/MambaDFuse-main", "third_party/MambaDFuse"],
    )
    for item in candidates:
        root = _path(item)
        if (root / "models" / "network.py").exists():
            return root
    return _path(candidates[0])


MAMBADFUSE_ROOT = _detect_mambadfuse_root()
MAMBADFUSE_MODEL_DIR = MAMBADFUSE_ROOT / _get(
    _fusion_conf,
    ["mambadfuse", "weights_dir"],
    "Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models",
)
MAMBADFUSE_ITER_NUMBER = str(_get(_fusion_conf, ["mambadfuse", "iter_number"], "10000"))

_MAMBADFUSE_PREFER_WEIGHT_TYPE = str(_get(_fusion_conf, ["mambadfuse", "prefer_weight_type"], "E"))
_MAMBADFUSE_FALLBACK_WEIGHT_TYPE = str(_get(_fusion_conf, ["mambadfuse", "fallback_weight_type"], "G"))

MAMBADFUSE_E_PATH = MAMBADFUSE_MODEL_DIR / f"{MAMBADFUSE_ITER_NUMBER}_E.pth"
MAMBADFUSE_G_PATH = MAMBADFUSE_MODEL_DIR / f"{MAMBADFUSE_ITER_NUMBER}_G.pth"

MAMBADFUSE_SCALE = int(_get(_fusion_conf, ["mambadfuse", "scale"], 1))
MAMBADFUSE_IN_CHANNEL = int(_get(_fusion_conf, ["mambadfuse", "in_channel"], 1))
MAMBADFUSE_OUT_CHANNEL = int(_get(_fusion_conf, ["mambadfuse", "out_channel"], 1))
MAMBADFUSE_IMG_SIZE = int(_get(_fusion_conf, ["mambadfuse", "img_size"], 128))
MAMBADFUSE_WINDOW_SIZE = int(_get(_fusion_conf, ["mambadfuse", "window_size"], 8))
MAMBADFUSE_EMBED_DIM = int(_get(_fusion_conf, ["mambadfuse", "embed_dim"], 60))
MAMBADFUSE_IMG_RANGE = float(_get(_fusion_conf, ["mambadfuse", "img_range"], 1.0))
MAMBADFUSE_MODEL_NAME = _get(_fusion_conf, ["mambadfuse", "model_name"], "MambaDFuse / Official")


# =========================================================
# 兼容旧 WEMFusion 字段名
# =========================================================

WEMFUSION_ROOT = MAMBADFUSE_ROOT
WEMFUSION_MODEL_DIR = MAMBADFUSE_ROOT / "models"

MODEL_DIR = MAMBADFUSE_MODEL_DIR
ITER_NUMBER = MAMBADFUSE_ITER_NUMBER
MODEL_TYPE = _MAMBADFUSE_FALLBACK_WEIGHT_TYPE
MODEL_PATH = MODEL_DIR / f"{ITER_NUMBER}_{MODEL_TYPE}.pth"
WEMFUSION_WEIGHT_PATH = MODEL_PATH

MODEL_NAME = _get(_fusion_conf, ["legacy_compat", "model_name"], "WEMFusion / Local-MambaDFuse")

SCALE = MAMBADFUSE_SCALE
IN_CHANNEL = MAMBADFUSE_IN_CHANNEL
OUT_CHANNEL = MAMBADFUSE_OUT_CHANNEL
IMG_SIZE = MAMBADFUSE_IMG_SIZE
WINDOW_SIZE = MAMBADFUSE_WINDOW_SIZE
EMBED_DIM = MAMBADFUSE_EMBED_DIM
IMG_RANGE = MAMBADFUSE_IMG_RANGE

IN_CHANNELS = IN_CHANNEL
OUT_CHANNELS = OUT_CHANNEL


# =========================================================
# 输入输出目录
# =========================================================

TEMP_DIR = ROOT_DIR / "data" / "temp"
OUTPUT_DIR = ROOT_DIR / "outputs"

for _dir in [TEMP_DIR, OUTPUT_DIR, KNOWLEDGE_DIR, RAG_INDEX_DIR, CHROMA_DIR, PROMPT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
