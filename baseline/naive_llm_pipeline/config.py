import os


OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))
