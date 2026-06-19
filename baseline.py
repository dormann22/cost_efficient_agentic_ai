from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

python_tool = PythonREPLTool()
tools = [python_tool]

model = init_chat_model(
    "openai:gpt-5.4",
    temperature=0,
    timeout=300,
    max_tokens=2500,
)

SYSTEM_PROMPT = """
You are a helpful and precise assistant for data preprocessing tasks.
You have access to a Python REPL tool that allows you to execute Python code and see the results.
When you need to perform data preprocessing, use the Python REPL tool to write and execute code that manipulates the dataset as needed.
"""

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

dataset_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

prompt = f"""
You are a data preprocessing agent. 
1. Load the dataset from this URL into a pandas DataFrame: {dataset_url}
2. Inspect the dataset to find out which columns have missing values.
3. Write and execute Python code to fill the missing numerical values with the mean, 
   and drop columns that have too many missing values (like 'Cabin').
4. Deal with any categorical variables by encoding them appropriately (e.g., using one-hot encoding).
4. Save the cleaned dataset as 'titanic_cleaned.csv' in the local directory.
5. Tell me a summary of what you did.
"""

inputs = {"messages": [HumanMessage(content=prompt)]}

for event in agent_executor.stream(inputs, stream_mode="values"):
    message = event["messages"][-1]
    
    # Print the agent's thoughts and actions
    message.pretty_print()



# FOR MONITORING LATER ON WITH LANGSMITH
#export LANGSMITH_TRACING="true"
#export LANGSMITH_API_KEY="..."