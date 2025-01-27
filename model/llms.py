from langchain_openai import ChatOpenAI
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from setup import SetupApi

def LLM(temperature: float = 0.6) -> ChatOpenAI:
    model = ChatOpenAI(
        api_key=SetupApi.open_ai_key,
        model_name="gpt-4o-mini",
        temperature=temperature,
        max_tokens=512*2
    )
    return model