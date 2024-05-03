from os import environ, getenv


def get_openai_api_key() -> str:
    OPENAI_API_KEY = "OPENAI_API_KEY"
    if OPENAI_API_KEY not in environ:
        raise ValueError(f"Please provide a {OPENAI_API_KEY}.")
    return getenv(OPENAI_API_KEY)
