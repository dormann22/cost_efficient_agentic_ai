import subprocess
import sys
from pathlib import Path
from langchain_core.tools import tool
from config import RUN_TIMEOUT_SECONDS, WORKSPACE_DIR

def _workspace() -> Path:
    workspace = Path(WORKSPACE_DIR)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace

def _resolve(filename: str) -> Path:
    # Keep all generated files inside the workspace
    return _workspace() / Path(filename).name

@tool
def write_code(filename: str, code: str) -> str:
    """Write Python code to a file in the agent workspace.

    Args:
        filename: Name of the script to create or overwrite, e.g. "preprocess.py".
        code: The full Python source code to write to the file.

    Returns:
        A confirmation message with the path the code was written to.
    """
    path = _resolve(filename)
    path.write_text(code, encoding="utf-8")
    return f"Wrote {len(code)} characters to {path}."


@tool
def run_code(filename: str) -> str:
    """Execute a Python script from the workspace and check whether it ran successfully.

    Args:
        filename: Name of the script to run, e.g. "preprocess.py".

    Returns:
        Whether the script succeeded, plus its stdout and stderr output.
    """
    path = _resolve(filename)
    if not path.exists():
        return f"FAILURE: {path} does not exist. Write the file first with write_code."

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"FAILURE: execution timed out after {RUN_TIMEOUT_SECONDS} seconds."

    status = "SUCCESS" if result.returncode == 0 else "FAILURE"
    return (
        f"{status} (exit code {result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


TOOLS = [write_code, run_code]
