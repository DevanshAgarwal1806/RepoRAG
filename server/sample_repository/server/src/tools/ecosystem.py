import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
import wikipedia
from langchain_core.tools import tool

from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.document_loaders import WikipediaLoader

from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.document_loaders import ArxivLoader

from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

from dotenv import load_dotenv

load_dotenv()

@tool
def wikipedia_full_read_tool(page_title: str) -> str:
    """
    Use this tool to read the FULL, un-truncated text of a Wikipedia article.
    Use this when you need deep, specific details, history, or facts from the page.
    Input should be the exact title of the Wikipedia page (e.g., 'Quantum computing').
    """
    try:
        # load_max_docs=1 ensures we only get the exact page requested
        loader = WikipediaLoader(query=page_title, load_max_docs=1, doc_content_chars_max=1000000)
        docs = loader.load()
        if docs:
            return docs[0].page_content
        return "Article not found. Try a different search term."
    except Exception as e:
        return f"Error loading full Wikipedia page: {e}"

@tool
def arxiv_full_read_tool(paper_id: str) -> str:
    """
    Use this tool to read the FULL text of an Arxiv paper. 
    Only use this AFTER you have found a specific paper ID using the search tool.
    Input should be the exact Arxiv ID (e.g., '1605.08386').
    """
    try:
        loader = ArxivLoader(query=paper_id, load_max_docs=1)
        docs = loader.load()
        if docs:
            return docs[0].page_content
        return "Paper not found."
    except Exception as e:
        return f"Error loading full paper: {e}"
    
@tool
def aviationstack_flights_tool(
    flight_iata: str = "",
    flight_icao: str = "",
    airline_iata: str = "",
    airline_name: str = "",
    flight_number: str = "",
    flight_date: str = "",
    dep_iata: str = "",
    arr_iata: str = "",
    limit: int = 5,
) -> str:
    """
    Query aviationstack flight data for a specific flight or route.
    Use this when the agent needs structured aviation data such as live flight status,
    departure and arrival airports, or scheduled/estimated times.
    Provide the most specific filters you have, such as flight_iata='AI302' or
    dep_iata='DEL' and arr_iata='SFO'. flight_date should be YYYY-MM-DD when used.
    """
    api_key = os.getenv("AVIATIONSTACK_API_KEY")
    if not api_key:
        return json.dumps(
            {"error": "AVIATIONSTACK_API_KEY is not set."},
            indent=2,
        )

    safe_limit = max(1, min(limit, 10))
    params = {
        "access_key": api_key,
        "limit": safe_limit,
    }

    optional_params = {
        "flight_iata": flight_iata,
        "flight_icao": flight_icao,
        "airline_iata": airline_iata,
        "airline_name": airline_name,
        "flight_number": flight_number,
        "flight_date": flight_date,
        "dep_iata": dep_iata,
        "arr_iata": arr_iata,
    }
    params.update({key: value for key, value in optional_params.items() if value})

    if len(params) <= 2:
        return json.dumps(
            {
                "error": (
                    "At least one search filter is required, such as flight_iata, "
                    "flight_number, dep_iata, or arr_iata."
                )
            },
            indent=2,
        )

    base_url = os.getenv("AVIATIONSTACK_BASE_URL", "http://api.aviationstack.com/v1")
    endpoint = f"{base_url.rstrip('/')}/flights?{urlencode(params)}"

    try:
        with urlopen(endpoint, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return json.dumps(
            {"error": f"Aviationstack request failed with HTTP {exc.code}."},
            indent=2,
        )
    except URLError as exc:
        return json.dumps(
            {"error": f"Could not reach aviationstack: {exc.reason}."},
            indent=2,
        )
    except Exception as exc:
        return json.dumps(
            {"error": f"Unexpected aviationstack error: {exc}."},
            indent=2,
        )

    if payload.get("error"):
        return json.dumps(payload["error"], indent=2)

    flights = []
    for item in payload.get("data", [])[:safe_limit]:
        flights.append(
            {
                "flight_date": item.get("flight_date"),
                "flight_status": item.get("flight_status"),
                "airline": {
                    "name": (item.get("airline") or {}).get("name"),
                    "iata": (item.get("airline") or {}).get("iata"),
                    "icao": (item.get("airline") or {}).get("icao"),
                },
                "flight": {
                    "number": (item.get("flight") or {}).get("number"),
                    "iata": (item.get("flight") or {}).get("iata"),
                    "icao": (item.get("flight") or {}).get("icao"),
                },
                "departure": {
                    "airport": (item.get("departure") or {}).get("airport"),
                    "iata": (item.get("departure") or {}).get("iata"),
                    "icao": (item.get("departure") or {}).get("icao"),
                    "scheduled": (item.get("departure") or {}).get("scheduled"),
                    "estimated": (item.get("departure") or {}).get("estimated"),
                    "actual": (item.get("departure") or {}).get("actual"),
                    "terminal": (item.get("departure") or {}).get("terminal"),
                    "gate": (item.get("departure") or {}).get("gate"),
                    "delay": (item.get("departure") or {}).get("delay"),
                },
                "arrival": {
                    "airport": (item.get("arrival") or {}).get("airport"),
                    "iata": (item.get("arrival") or {}).get("iata"),
                    "icao": (item.get("arrival") or {}).get("icao"),
                    "scheduled": (item.get("arrival") or {}).get("scheduled"),
                    "estimated": (item.get("arrival") or {}).get("estimated"),
                    "actual": (item.get("arrival") or {}).get("actual"),
                    "terminal": (item.get("arrival") or {}).get("terminal"),
                    "gate": (item.get("arrival") or {}).get("gate"),
                    "baggage": (item.get("arrival") or {}).get("baggage"),
                    "delay": (item.get("arrival") or {}).get("delay"),
                },
                "live": item.get("live"),
            }
        )

    return json.dumps(
        {
            "request_filters": {key: value for key, value in optional_params.items() if value},
            "count": len(flights),
            "flights": flights,
        },
        indent=2,
    )


def get_research_tools():
    """Returns a list of tools for academic and general research (Skimmers + Deep Readers)."""
    # We pass wiki_client=wikipedia to bypass the Pydantic validation bug
    wiki_wrapper = WikipediaAPIWrapper(wiki_client=wikipedia, top_k_results=3, doc_content_chars_max=2000)
    wiki_search_tool = WikipediaQueryRun(
        api_wrapper=wiki_wrapper,
        description="Search Wikipedia for summaries and page titles. Use this FIRST to find the exact page title."
    )
    
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
    arxiv_search_tool = ArxivQueryRun(
        api_wrapper=arxiv_wrapper,
        description="Search Arxiv for abstracts and Paper IDs. Use this FIRST to find relevant paper IDs."
    )
    
    # Return both the skimmers and our custom deep readers
    return [
        wiki_search_tool, 
        wikipedia_full_read_tool, 
        arxiv_search_tool, 
        arxiv_full_read_tool,
        aviationstack_flights_tool
    ]