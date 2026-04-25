import os
from dotenv import load_dotenv

load_dotenv()

# ── Named keys for specific roles ─────────────────────────────────────────────
GROQ_API_KEY_GENERATOR = os.getenv("GROQ_API_KEY_GENERATOR")
GROQ_API_KEY_EVALUATOR = os.getenv("GROQ_API_KEY_EVALUATOR")

# ── Pool of 8 extra keys for high-frequency calls ────────────────────────────
# (tool selector + generalizer fire on every single task, so they need rotation)
GROQ_KEY_POOL = [
    key for key in [
        os.getenv("GROQ_API_KEY_3"),
        os.getenv("GROQ_API_KEY_4"),
        os.getenv("GROQ_API_KEY_5"),
        os.getenv("GROQ_API_KEY_6"),
        os.getenv("GROQ_API_KEY_7"),
        os.getenv("GROQ_API_KEY_8"),
        os.getenv("GROQ_API_KEY_9"),
        os.getenv("GROQ_API_KEY_10"),
    ]
    if key  # silently skip any that aren't set
]

# ── Startup validation ────────────────────────────────────────────────────────
if not GROQ_API_KEY_GENERATOR:
    raise EnvironmentError("GROQ_API_KEY_GENERATOR is not set in your .env file")
if not GROQ_API_KEY_EVALUATOR:
    raise EnvironmentError("GROQ_API_KEY_EVALUATOR is not set in your .env file")
if not GROQ_KEY_POOL:
    raise EnvironmentError(
        "No keys found in pool (GROQ_API_KEY_3 through GROQ_API_KEY_10). "
        "At least one must be set."
    )

print(f"[settings] Key pool loaded: {len(GROQ_KEY_POOL)} keys available")

# ── Tuning constants ──────────────────────────────────────────────────────────
DAG_MAX_RETRIES               = 3
DAG_PASSING_SCORE             = 13
TASK_MAX_RETRIES              = 2
TOOL_MATCH_THRESHOLD          = 7
TOOL_MAX_GENERALIZATION_STEPS = 4