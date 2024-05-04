import logging
import os
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="app.log",  # Logs will be saved to 'app.log'
    filemode="w",  # 'w' for overwrite (use 'a' for append)
)

logger = logging.getLogger(__name__)

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

# Prepare the model and tokenizer if CUDA is available
if torch.cuda.is_available():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
else:
    logger.error("CUDA is not available. This script requires a GPU to run.")
    exit(1)


def generate(queries: List[str], max_new_tokens: int = 1024) -> List[str]:
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
    queries = [
        "What is the impact of AI on society?",
        "Tell me about the latest advancements in machine learning.",
        "How does quantum computing affect data security?",
    ]
    responses = generate(queries)
    for response in responses:
        print(response)
