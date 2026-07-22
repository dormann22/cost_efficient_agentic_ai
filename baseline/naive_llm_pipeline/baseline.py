from pathlib import Path
from graph import build_graph

def main() -> None:
    app = build_graph()
    initial_state = {
        "dataset_path": str(Path("../data") / "sample_titanic.csv"),
        "target_task": "Perform data preprocessing on this CSV chunk.",
        "chunk_size_chars": 12000,
        "max_chunks": 4,
    }
    print("=== Building Graph ===")

    result = app.invoke(initial_state)
    print("=== Final Naive Baseline Output ===")
    print(result["final_report"])

if __name__ == "__main__":
    main()