import os

OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))

# Max number of agent <-> tool turns before the graph stops
RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "25"))

#Where the agent writes and runs the generated code
WORKSPACE_DIR = os.getenv("AGENT_WORKSPACE_DIR", "generated")

# Timeout for executing code
RUN_TIMEOUT_SECONDS = int(os.getenv("AGENT_RUN_TIMEOUT_SECONDS", "60"))

# Where run reports (steps taken + token usage) are written
LOG_DIR = os.getenv("AGENT_LOG_DIR", "logs")
