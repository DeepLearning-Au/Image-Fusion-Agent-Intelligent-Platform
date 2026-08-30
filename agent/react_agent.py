from pathlib import Path
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.tools.agent_tools import build_wemfusion_tools
from config import settings
from model.factory import get_chat_model


def load_prompt_file(path, default_text: str = "") -> str:
    """
    从 prompts 目录读取提示词文件。
    这样以后修改 Agent 行为时，只需要改 txt 文件，不需要改 Python 代码。
    """
    path = Path(path)

    if not path.exists():
        return default_text

    return path.read_text(encoding="utf-8")


def convert_history(history: Optional[List[Dict]]):
    """
    将 Streamlit 中的历史消息转换为 LangChain 消息格式。
    用于支持多轮对话。
    """
    messages = []

    for item in history or []:
        role = item.get("role")
        content = item.get("content", "")

        if not content:
            continue

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def build_agent_executor(
    last_result: Optional[Dict] = None,
    last_report: Optional[str] = None,
    optical_path: Optional[str] = None,
    sar_path: Optional[str] = None,
    use_tile: bool = False,
    tile: int = 256,
    overlap: int = 32,
    user_city: str = "烟台",
    tool_state: Optional[Dict] = None,
):
    """
    构建 Qwen Tool Calling Agent。

    提示词从 prompts/main_prompt.txt 读取，
    不再写死在 Python 文件里。
    """
    llm = get_chat_model()

    tools = build_wemfusion_tools(
        last_result=last_result,
        last_report=last_report,
        optical_path=optical_path,
        sar_path=sar_path,
        use_tile=use_tile,
        tile=tile,
        overlap=overlap,
        user_city=user_city,
        tool_state=tool_state,
    )

    main_prompt = load_prompt_file(
        settings.MAIN_PROMPT_PATH,
        default_text=(
            "你是 MambaDFuse-Agent，一个面向红外、可见光与双模态成像设备的智能选型和融合助手。"
            "你需要根据用户问题自动选择工具，并基于工具结果回答。"
        ),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", main_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )

    return executor


def run_langchain_agent(
    question: str,
    history: Optional[List[Dict]] = None,
    last_result: Optional[Dict] = None,
    last_report: Optional[str] = None,
    optical_path: Optional[str] = None,
    sar_path: Optional[str] = None,
    use_tile: bool = False,
    tile: int = 256,
    overlap: int = 32,
    user_city: str = "烟台",
):
    """
    运行大模型 Agent。

    返回：
    - answer：大模型最终回答；
    - trace：工具调用轨迹；
    - tool_state：工具产生的新状态，例如融合结果和报告。
    """
    tool_state = {}

    executor = build_agent_executor(
        last_result=last_result,
        last_report=last_report,
        optical_path=optical_path,
        sar_path=sar_path,
        use_tile=use_tile,
        tile=tile,
        overlap=overlap,
        user_city=user_city,
        tool_state=tool_state,
    )

    response = executor.invoke(
        {
            "input": question,
            "chat_history": convert_history(history or []),
        }
    )

    output = response.get("output", "")
    steps = response.get("intermediate_steps", [])

    trace = []

    for step in steps:
        try:
            action, observation = step

            trace.append(
                {
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "observation": str(observation)[:2200],
                }
            )

        except Exception:
            trace.append(
                {
                    "tool": "unknown",
                    "tool_input": "",
                    "observation": str(step)[:2200],
                }
            )

    return {
        "answer": output,
        "trace": trace,
        "tool_state": tool_state,
    }
