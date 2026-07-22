from pathlib import Path

from langchain_core.messages import HumanMessage

from config import RECURSION_LIMIT
from graph import build_graph
from prompts import build_task_prompt
from reporting import build_run_report, write_run_report


def main() -> None:
    app = build_graph()

    dataset_path = str(Path("../data") / "sample_titanic.csv")
    target_task = "Perform data preprocessing on this CSV."
    task_prompt = build_task_prompt(dataset_path, target_task)

    initial_state = {
        "dataset_path": dataset_path,
        "target_task": target_task,
        "messages": [HumanMessage(content=task_prompt)],
    }

    print("=== Building Graph ===")
    result = app.invoke(initial_state, {"recursion_limit": RECURSION_LIMIT})

    print("=== Final Naive Agent Output ===")
    print(result["messages"][-1].content)

    print("\n" + build_run_report(result["messages"]))
    log_path = write_run_report(result["messages"])
    print(f"\nRun report written to {log_path}")


if __name__ == "__main__":
    main()
