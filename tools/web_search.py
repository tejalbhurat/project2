import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the web with Tavily and return source-rich results."""
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is not configured in .env")

    try:
        search = TavilySearch(max_results=max_results)
        response = search.invoke({"query": query})
        if isinstance(response, dict) and "results" in response:
            return response["results"]
        if isinstance(response, list):
            return response
        return []
    except Exception as exc:
        raise RuntimeError(f"Tavily search failed for {query!r}: {exc}") from exc
