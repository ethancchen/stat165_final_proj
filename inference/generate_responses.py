import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from openai import AsyncOpenAI
from pandas import Index
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from tqdm import tqdm

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.helpers import get_openai_api_key  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="app.log",
    filemode="w",
)

logger = logging.getLogger(__name__)

MAX_MAX_NEW_TOKENS = 16000
# DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
DPO_PATH = ""
CUSTOM_CACHE_DIR = ""
GENERAL_PROMPT = "general_prompt"
LLAMA_RESPONSE = "llama_response"
GPT35_RESPONSE = "gpt35_response"
QUESTION = "question"
RESOLUTION_CRITERIA = "resolution_criteria"

INPUT_FILEPATH = ""
OUTPUT_PATH = ""


class Inferencer:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        assert len(self.df) == 100, "test set must have 100 rows of forecasting data"
        self.df =self.df.head(10)
        self.general_prompt_template = """Here is a forecasting question: {question}\n\nHere are the resolution criteria of the question: {resolution_criteria}.\n\nBased on the forecasting question and resolution criteria, generate step-by-step reasoning to create a resolution. Output exactly 6 steps without an introduction or a conclusion."""  # noqa: E501
        # self.client = AsyncOpenAI(api_key=get_openai_api_key())

    def __len__(self) -> int:
        return len(self.df)

    def _get_df_columns(self) -> Index:
        return self.df.columns

    def format_general_prompt(self, question: str, resolution_criteria: str) -> str:
        assert len(question) > 0, f"Please provide a non-empty string for the {QUESTION}."
        assert len(resolution_criteria) > 0, f"Please provide a non-empty string for the {RESOLUTION_CRITERIA}."
        return self.general_prompt_template.format(
            question=question,
            resolution_criteria=resolution_criteria,
        )

    def populate_df_general_prompts(self) -> None:
        assert GENERAL_PROMPT not in self._get_df_columns()
        self.df[GENERAL_PROMPT] = self.df.apply(
            lambda row: self.format_general_prompt(row[QUESTION], row[RESOLUTION_CRITERIA]), axis=1
        )

    def get_all_llama_responses(self, model, tokenizer):
        assert LLAMA_RESPONSE not in self._get_df_columns()
        text_generator = TextGenerationPipeline(model, tokenizer, truncation=True)
        llama_responses = []
        rows = [row for _, row in self.df.iterrows()]
        for row in tqdm(rows):
            prompt = row[GENERAL_PROMPT]
            output = text_generator(prompt, max_length=4096,truncation=True)
            response = output[0]["generated_text"]
            llama_responses.append(response)

        self.df.loc[:, LLAMA_RESPONSE] = llama_responses
        output_filepath = OUTPUT_PATH
        self.df.to_csv(output_filepath, index=False)

    async def prompt_gpt35_once(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125", messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing prompt: {prompt}\nError: {e}")
            return "Error in response"  # Placeholder or handle appropriately

    async def get_all_gpt35_responses(self) -> None:
        """Call this only once. DO NOT call this along with every single LlaMA model generation."""
        assert GPT35_RESPONSE not in self._get_df_columns()

        # Create a list of tasks for asynchronous execution
        tasks = [self.prompt_gpt35_once(row[GENERAL_PROMPT]) for _, row in self.df.iterrows()]
        gpt35_responses = await asyncio.gather(*tasks)

        self.df[GPT35_RESPONSE] = gpt35_responses
        output_filepath = OUTPUT_PATH
        self.df.to_csv(output_filepath, index=False)


async def main():
    parser = argparse.ArgumentParser(description="Generate responses based on model type.")
    parser.add_argument(
        "--is_gpt",
        choices=["True", "False"],
        required=False,
        help='A boolean. "True" for using GPT, "False" otherwise. Case-sensitive and required.',
    )
    args = parser.parse_args()
    is_gpt = args.is_gpt
    inferencer = Inferencer(data_path=INPUT_FILEPATH)
    inferencer.populate_df_general_prompts()
    if is_gpt:
        await inferencer.get_all_gpt35_responses()
    else:
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=CUSTOM_CACHE_DIR)
            model.max_position_embeddings = 32768
            model.resize_token_embeddings(128258)
            # model.load_adapter(DPO_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, truncation=True)
        else:
            logger.error("CUDA is not available. This script requires a GPU to run.")
            exit(1)
        inferencer.get_all_llama_responses(model, tokenizer)


if __name__ == "__main__":
    # client = OpenAI(api_key=get_openai_api_key())
    # prompt = "What is 1+1?"
    # try:
    #     response = client.chat.completions.create(
    #         model="gpt-3.5-turbo-0125", messages=[{"role": "user", "content": prompt}]
    #     )
    #     print(response.choices[0].message.content)
    # except Exception as e:
    #     print(f"Error processing prompt: {prompt}\nError: {e}")
    #     print("Error in response")  # Placeholder or handle appropriately

    asyncio.run(main())