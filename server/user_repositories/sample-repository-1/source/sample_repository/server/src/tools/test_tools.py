import os
import json
from dotenv import load_dotenv

from src.tools.openapi_parser import get_openapi_tools

# Load environment variables (API keys)
load_dotenv()

# Import the tools you built
from src.tools import get_research_tools
from src.tools import get_search_tool
from src.tools import firecrawl_scrape

def save_to_file(filename, content, is_json=False):
    """Helper function to save massive text outputs to a file."""
    # Create a test_results directory if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    filepath = os.path.join("test_results", filename)
    
    if is_json:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Saved full output to: {filepath}")

def test_tavily_tool():
    print("--- Testing Tavily Web Search ---")
    if not os.getenv("TAVILY_API_KEY"):
        print("SKIP: TAVILY_API_KEY not found in .env file.\n")
    else:
        try:
            search_tools = get_search_tool()
            tavily_tool = search_tools[0]
            tavily_query = "Latest breakthrough in Deep Reinforcement Learning"
            print(f"Searching the live web for: '{tavily_query}'...")
            tavily_result = tavily_tool.invoke({"query": tavily_query})
            print("Success! Web Search Results retrieved.")
            save_to_file("tavily_search_results.json", tavily_result, is_json=True)
            print()
        except Exception as e:
            print(f"Tavily Search Failed: {e}\n")
            
def test_wikipedia_and_arxiv_tools():
    print("Loading research tools...")
    research_tools = get_research_tools()
    wiki_search, wiki_deep_read, arxiv_search, arxiv_deep_read = research_tools

    # --- Wikipedia Tests ---
    print("\n--- Testing Wikipedia Tools ---")
    wiki_query = "Alan Turing"
    
    # Test Skimmer
    try:
        print("1. Testing Skimmer...")
        wiki_skimmer_result = wiki_search.invoke({"query": wiki_query})
        print(f"Skimmer Success! Snippet: {wiki_skimmer_result[:100]}...")
    except Exception as e:
        print(f"Wikipedia Skimmer Failed: {e}")

    # Test Deep Reader
    try:
        print("2. Testing Deep Reader...")
        # Note: The input key matches the variable name in your @tool definition
        wiki_deep_result = wiki_deep_read.invoke({"page_title": wiki_query})
        print(f"Deep Reader Success! Retrieved {len(wiki_deep_result)} characters.")
        save_to_file("wikipedia_alan_turing_full.txt", wiki_deep_result)
    except Exception as e:
        print(f"Wikipedia Deep Reader Failed: {e}")

    # --- Arxiv Tests ---
    print("\n--- Testing Arxiv Tools ---")
    # '1706.03762' is the famous "Attention Is All You Need" paper ID
    arxiv_paper_id = "1706.03762" 
    
    # Test Skimmer
    try:
        print("1. Testing Skimmer...")
        arxiv_skimmer_result = arxiv_search.invoke({"query": "Attention is all you need"})
        print(f"Skimmer Success! Snippet: {arxiv_skimmer_result[:100]}...")
    except Exception as e:
        print(f"Arxiv Skimmer Failed: {e}")

    # Test Deep Reader
    try:
        print("2. Testing Deep Reader...")
        arxiv_deep_result = arxiv_deep_read.invoke({"paper_id": arxiv_paper_id})
        print(f"Deep Reader Success! Retrieved {len(arxiv_deep_result)} characters.")
        save_to_file("arxiv_attention_paper_full.txt", arxiv_deep_result)
    except Exception as e:
        print(f"Arxiv Deep Reader Failed: {e}")

def test_openapi_tool():
    print("\n--- Testing OpenAPI Tools ---")
    try:
        openapi_tools = get_openapi_tools()
        if not openapi_tools:
            print("No OpenAPI tools generated. Please add a valid OpenAPI spec to data/openapi_specs/ and try again.")
            return
        print(f"Success! Generated {len(openapi_tools)} OpenAPI tools.")
        for tool in openapi_tools:
            print(f"- {tool.name}: {tool.description}")
    except Exception as e:
        print(f"OpenAPI Tools Test Failed: {e}")
        
def test_firecrawl_tool():
    print("\n--- Testing Firecrawl Scrape Tool ---")
    try:
        url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        result = firecrawl_scrape.invoke({"url": url})
        print(f"Success! Scraped content length: {len(result)} characters.")
        save_to_file("firecrawl_wikipedia_ai.md", result)
    except Exception as e:
        print(f"Firecrawl Scrape Tool Failed: {e}")

if __name__ == "__main__":
    # test_tavily_tool()
    # test_wikipedia_and_arxiv_tools()
    # test_openapi_tool()
    test_firecrawl_tool()