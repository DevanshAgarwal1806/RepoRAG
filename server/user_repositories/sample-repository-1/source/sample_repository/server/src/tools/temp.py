import os
from dotenv import load_dotenv

load_dotenv()

print("FIRECRAWL KEY:", os.getenv("FIRECRAWL_API_KEY"))