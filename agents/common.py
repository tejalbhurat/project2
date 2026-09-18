import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(override=True)


def get_llm(temperature: float = 0.0, max_tokens: int = 1400) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured in .env")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_format="hidden",
        reasoning_effort="low",
        api_key=api_key,
    )
