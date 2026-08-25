"""Demo entry point for the token/cost tracer (src/agentprep/token_tracking.py).

No real agent exists in src/agentprep yet, so this script exercises the
tracer against a small synthetic conversation standing in for a LangGraph
run's `result["messages"]`. Once a real agent lands, call
`extract_token_usage(result["messages"], config.cost.pricing)` directly from
its own run script instead of this demo.

Run with: python scripts/token_report.py
"""

from langchain_core.messages import AIMessage, HumanMessage

from agentprep.config import load_config
from agentprep.token_tracking import extract_token_usage, write_token_report


def _demo_messages() -> list:
    """Stand-in for a real agent's result["messages"]."""
    return [
        HumanMessage(content="Preprocess this dataset."),
        AIMessage(
            content="Let me inspect the schema first.",
            usage_metadata={"input_tokens": 850, "output_tokens": 120, "total_tokens": 970},
            response_metadata={"model_name": "gpt-4o-mini"},
        ),
        AIMessage(
            content="This needs the bigger model to reason about outliers.",
            usage_metadata={"input_tokens": 2400, "output_tokens": 300, "total_tokens": 2700},
            response_metadata={"model_name": "gpt-4o"},
        ),
    ]


def main() -> None:
    config = load_config()
    messages = _demo_messages()

    report = extract_token_usage(messages, pricing=config.cost.pricing)
    json_path, csv_path = write_token_report(report, config.path("logs"), run_label="demo")

    print("=== Token Report (demo) ===")
    for model_name, usage in report.by_model.items():
        print(
            f"{model_name}: calls={usage.llm_calls} "
            f"in={usage.input_tokens} out={usage.output_tokens} "
            f"total={usage.total_tokens} cost=${usage.cost_usd:.4f}"
        )
    print(
        f"TOTAL: calls={report.total_llm_calls} "
        f"in={report.total_input_tokens} out={report.total_output_tokens} "
        f"total={report.total_tokens} cost=${report.total_cost_usd:.4f}"
    )
    print(f"\nDetail written to {json_path}")
    print(f"Ledger row appended to {csv_path}")


if __name__ == "__main__":
    main()
