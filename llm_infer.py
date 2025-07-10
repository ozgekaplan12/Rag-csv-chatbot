from langchain_community.llms import Ollama
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

def get_llm():
    return Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0.1)