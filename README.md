# agentprep

Cost-efficient agentic AI for tabular data preprocessing.

This MSC thesis project explores how an agentic workflow can preprocess datasets in the most cost-efficient way while also keeping performance at an acceptable level, balancing this tradeoff most efficiently. It utilizes **metadata** to describe datasets. The agent uses this metadata to plan out and execute the preprocessing steps instead of reading the whole dataset. This way we can lower the amount of tokens that are fed into the actual LLM models and by doing this we also decrease the cost of the whole preprocessing process done by the agentic flow.
Also by **dynamic routing** the agent tries to adapt to the difficulty of the individual tasks and use cheaper, on-prem models for easier sub-tasks, only relying on big commercial LLM API-s for complex tasks.

Setup:
```bash
# 1. Clone, then from the repo root:
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the dependency set + the package.
pip install -r requirements.txt
pip install -e .