SYSTEM_PROMPT = """
You are a data preprocessing assistant.
You receive large raw data chunks directly.
Analyze the raw text chunk and propose preprocessing actions.
""".strip()


def build_chunk_prompt(chunk: str, target_task: str, chunk_index: int, total_chunks: int) -> str:
    return (
        f"Task: {target_task}\n"
        f"Chunk {chunk_index}/{total_chunks}\n"
        "You are given raw CSV text. Identify issues and propose preprocessing steps.\n"
        "Return:\n"
        "1) detected problems\n"
        "2) concrete cleaning operations\n"
        "3) assumptions\n"
        "\n"
        "RAW CHUNK START\n"
        f"{chunk}\n"
        "RAW CHUNK END\n"
    )
