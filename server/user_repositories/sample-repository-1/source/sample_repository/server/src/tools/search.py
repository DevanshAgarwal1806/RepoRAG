import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_classic.agents import tool
from langchain_tavily import TavilySearch

load_dotenv()
firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

@tool
def firecrawl_scrape(url: str) -> str:
    """
    Scrape a webpage and return clean markdown content.
    Use this when the agent needs to read the full content of a webpage.
    Input should be a full URL.
    """
    
    result = firecrawl.scrape(
        url,
        formats=["markdown"]
    )

    return result.markdown

def get_search_tools():
    """
    Initializes the Tavily Search tool for agent-optimized web intelligence.
    The LLM will use this when it needs to find real-time, up-to-date information.
    """
    # max_results limits the output so we don't overwhelm the agent's memory
    search_tool = TavilySearch(max_results=10, api_key=os.getenv("TAVILY_API_KEY"))
    
    return [search_tool, firecrawl_scrape]