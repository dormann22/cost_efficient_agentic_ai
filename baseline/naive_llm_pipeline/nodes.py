from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config import OPENAI_MODEL, OPENAI_API_KEY, TEMPERATURE
from prompts import SYSTEM_PROMPT, build_chunk_prompt
from state import BaselineState

def load_raw_data(state: BaselineState) -> BaselineState:
    dataset_path = Path(state["dataset_path"])
    raw_data = dataset_path.read_text(encoding="utf-8")
    return {"raw_data": raw_data}

def split_into_large_chunks(state: BaselineState) -> BaselineState:
    raw_data = state["raw_data"]
    chunk_size = state.get("chunk_size_chars", 12000)
    max_chunks = state.get("max_chunks", 4)

    chunks = [raw_data[i : i + chunk_size] for i in range(0, len(raw_data), chunk_size)]
    chunks = chunks[:max_chunks]
    return {"chunks": chunks}

def process_chunks_with_openai(state: BaselineState) -> BaselineState:
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=TEMPERATURE,
    )

    chunks = state["chunks"]
    target_task = state["target_task"]
    outputs = []

    for idx, chunk in enumerate(chunks, start=1):
        prompt = build_chunk_prompt(
            chunk=chunk,
            target_task=target_task,
            chunk_index=idx,
            total_chunks=len(chunks),
        )
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        outputs.append(response.content)

    return {"chunk_outputs": outputs}

def aggregate_outputs(state: BaselineState) -> BaselineState:
    outputs = state.get("chunk_outputs", [])
    lines = ["Naive preprocessing baseline summary:"]
    for idx, output in enumerate(outputs, start=1):
        lines.append(f"\n--- Chunk {idx} ---")
        lines.append(str(output))

    final_report = "\n".join(lines)
    return {"final_report": final_report}
