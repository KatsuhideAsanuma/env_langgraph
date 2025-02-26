"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model
from langchain_core.output_parsers import StrOutputParser
# Define the function that calls the model


async def call_model(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.
    
    If the user asks to 'summarize this conversation', please respond with exactly that phrase so I can provide a summary.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# 要約機能を行う新しいノードを追加
def summarize(state: State) -> Dict[str, List[AIMessage]]:
    """Summarize the conversation so far."""
    messages = state.messages
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages if hasattr(msg, 'content') and msg.content])
    
    llm = load_chat_model("anthropic/claude-3-5-sonnet")
    prompt = f"Summarize this conversation briefly:\n{chat_history}"
    summary_text = StrOutputParser().invoke(llm.invoke(prompt))
    
    # 要約結果をAIMessageとして返す
    summary_message = AIMessage(content=f"Here's a summary of our conversation: {summary_text}")
    return {"messages": [summary_message]}

# グラフ定義 - 1つのbuilderのみ使用
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# 既存のノードを追加
builder.add_node("agent", call_model)  # 名前を明示的に"agent"に変更
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("summarize", summarize)  # 新しいノードを追加

# エントリーポイントを設定
builder.add_edge("__start__", "agent")

# モデル出力のルーティング関数
def route_model_output(state: State) -> Literal["__end__", "tools", "summarize"]:
    """Determine the next node based on the model's output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    
# 要約を明示的に要求した場合のルーティング
    if isinstance(last_message.content, str) and any(phrase in last_message.content.lower() 
                                               for phrase in ["summarize", "summary", "要約"]):
        return "summarize"
    
    # ツール呼び出しがあればツールノードへ
    if last_message.tool_calls:
        return "tools"
    
    # それ以外は終了
    return "__end__"

# 条件付きエッジを追加
builder.add_conditional_edges(
    "agent",
    route_model_output,
)

# ツールから再度エージェントへ
builder.add_edge("tools", "agent")

# 要約から再度エージェントへ
builder.add_edge("summarize", "agent")

# グラフのコンパイル
graph = builder.compile()
graph.name = "ReAct Agent with Summarization"  # 名前を更新