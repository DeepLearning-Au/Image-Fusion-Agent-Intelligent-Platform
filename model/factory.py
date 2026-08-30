from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

from config import settings


def get_chat_model():
    """
    获取 Qwen 大模型。

    作用：
    1. 作为 LangChain Agent 的大脑；
    2. 根据用户问题判断是否调用工具；
    3. 负责综合 RAG 检索结果、融合指标、天气结果和工具返回内容；
    4. 生成最终自然语言回答。
    """
    if not settings.DASHSCOPE_API_KEY:
        raise ValueError(
            "没有检测到 DASHSCOPE_API_KEY。\n"
            "请先在 PowerShell 中执行：\n"
            "$env:DASHSCOPE_API_KEY='你的阿里云百炼API Key'"
        )

    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.DASHSCOPE_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
    )

    return llm


def get_embedding_model():
    """
    获取 DashScope Embedding 模型。

    作用：
    1. 将产品说明书、故障案例、维护手册等知识库文档转成向量；
    2. 支持本地 RAG 检索；
    3. 后续由 Agent 调用 rag_search_tool 检索知识库内容；
    4. 避免使用 Chroma 默认 ONNX embedding，减少 onnxruntime 依赖问题。
    """
    if not settings.DASHSCOPE_API_KEY:
        raise ValueError(
            "没有检测到 DASHSCOPE_API_KEY，无法调用 DashScope Embedding。\n"
            "请先在 PowerShell 中执行：\n"
            "$env:DASHSCOPE_API_KEY='你的阿里云百炼API Key'"
        )

    embedding = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )

    return embedding
