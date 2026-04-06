import requests
import functools

def retry(max_attempts=3):
    def decorator(func):
        @functools.wraps(func)  # decorator call: functools.wraps
        def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    pass
        return wrapper
    return decorator

@retry(max_attempts=5)       # decorator call: retry
def fetch_data(url):
    response = requests.get(url)   # body call: requests.get
    return response.json()         # body call: response.json