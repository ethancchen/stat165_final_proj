import logging
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="app.log",
    filemode="w",
)

logger = logging.getLogger(__name__)

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
CUSTOM_CACHE_DIR = Path(
    "/", "global", "scratch", "users", "ethancchen", ".cache", "huggingface", "models--meta-llama--Meta-Llama-3-8B"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def generate(queries: list[str], max_new_tokens: int = 1024) -> list[str]:
    responses = []
    text_generator = TextGenerationPipeline(model, tokenizer)

    for query in queries:
        logger.info(f"Processing query: {query}")
        output = text_generator(query, max_length=MAX_INPUT_TOKEN_LENGTH + max_new_tokens)
        response = output[0]["generated_text"]
        responses.append(response)
        logger.info(f"Generated response: {response}")

    return responses


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto", cache_dir=CUSTOM_CACHE_DIR
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CUSTOM_CACHE_DIR)
    else:
        logger.error("CUDA is not available. This script requires a GPU to run.")
        exit(1)
    queries = [
        "What is the impact of AI on society?",
        "Tell me about the latest advancements in machine learning.",
        "How does quantum computing affect data security?",
    ]
    responses = generate(queries)
    for response in responses:
        print(response)
    output_filepath = OUTPUT_DIR / f"{MODEL_ID}_responses.csv"
    pd.DataFrame({"query": queries, "response": responses}).to_csv(output_filepath, index=False)
