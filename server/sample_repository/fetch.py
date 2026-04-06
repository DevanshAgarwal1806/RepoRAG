# server/sample_repository/fetch.py
import asyncio
import functools

def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    pass
        return wrapper
    return decorator

@retry(max_attempts=5)
async def fetch_data(url):
    """Fetches data from the given URL asynchronously."""
    reader, writer = await asyncio.open_connection(url, 80)
    writer.write(b"GET / HTTP/1.0\r\n\r\n")
    data = await reader.read(100)
    writer.close()
    return data

async def process_batch(items):
    results = await asyncio.gather(*[fetch_data(item) for item in items])
    return results

if __name__ == "__main__":
    asyncio.run(process_batch(["example.com", "test.com"]))