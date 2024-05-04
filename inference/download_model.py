from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model() -> None:
    custom_cache_dir = Path("/", "global", "scratch", "users", "ethancchen", ".cache", "huggingface")
    AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=custom_cache_dir)
    AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=custom_cache_dir)


if __name__ == "__main__":
    download_model()
