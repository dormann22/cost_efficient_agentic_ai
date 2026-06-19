from typing import List, TypedDict


class BaselineState(TypedDict, total=False):
    dataset_path: str
    target_task: str
    chunk_size_chars: int
    max_chunks: int
    raw_data: str
    chunks: List[str]
    chunk_outputs: List[str]
    final_report: str
