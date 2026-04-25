# src/tools/registry.py

from src.tools.search import get_search_tools
from src.tools.ecosystem import get_research_tools
from src.tools.openapi_parser import get_openapi_tools


def get_all_tools():
    """Single source of truth for all available tools."""
    return get_search_tools() + get_research_tools() 
    # return get_search_tools() + get_research_tools() + get_openapi_tools()


def build_tool_manifest() -> str:
    """
    Builds a concise tool listing to inject into the generator prompt.
    Format: one tool per line → "- tool_name: description"
    """
    tools = get_all_tools()
    lines = [f"- {t.name}: {t.description}" for t in tools]
    return "\n".join(lines)