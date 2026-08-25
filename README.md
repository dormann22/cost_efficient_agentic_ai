# agentprep

Cost-efficient agentic AI for tabular data preprocessing.

This MSC thesis project explores how an agentic workflow can preprocess datasets in the most cost-efficient way while also keeping performance at an acceptable level, balancin. It utilizes **metadata** to describe the datasets rather than reading the whole dataset, this way lowering the amount of tokens fed into the models
token usage, thus the cost of the preprocessing
Also by **dynamic routing** the agent uses cheaper models for easier sub-tasks, only relying on big API LLM-s for complex tasks.

Setup:
```bash
# 1. Clone, then from the repo root:
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the dependency set + the package.
pip install -r requirements.txt
pip install -e .