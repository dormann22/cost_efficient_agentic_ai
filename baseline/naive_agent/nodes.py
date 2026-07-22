from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE
from prompts import SYSTEM_PROMPT
from state import AgentState
from tools import TOOLS

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=TEMPERATURE,
).bind_tools(TOOLS)

def agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    return {"messages": [response]}
