import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from openai import OpenAI
from pandas import Index
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.helpers import get_openai_api_key  # noqa: E402

# Setup logging
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
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
CUSTOM_CACHE_DIR = Path(
    "/", "global", "scratch", "users", "ethancchen", ".cache", "huggingface", "models--meta-llama--Meta-Llama-3-8B"
)
GENERAL_PROMPT = "general_prompt"
LLAMA_RESPONSE = "llama_response"
QUESTION = "question"
RESOLUTION_CRITERIA = "resolution_criteria"

INPUT_FILEPATH = Path(__file__).resolve().parents[1] / "data" / "sampled_100_test_hf_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class Inferencer:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.general_prompt_template = """Here is a forecasting question: {question}\n\nHere are the resolution criteria of the question: {resolution_criteria}.\n\nBased on the forecasting question and resolution criteria, generate step-by-step reasoning to create a resolution. Output exactly 6 steps without an introduction or a conclusion."""  # noqa: E501
        self.client = OpenAI(api_key=get_openai_api_key())

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

    def get_all_llama_responses(self):
        assert LLAMA_RESPONSE not in self._get_df_columns()
        text_generator = TextGenerationPipeline(model, tokenizer)
        llama_responses = []

        for _, row in self.df.iterrows():
            query = row[GENERAL_PROMPT]
            output = text_generator(query, max_length=MAX_INPUT_TOKEN_LENGTH + MAX_MAX_NEW_TOKENS)
            response = output[0]["generated_text"]
            llama_responses.append(response)

        self.df.loc[:, LLAMA_RESPONSE] = llama_responses
        output_filepath = OUTPUT_DIR / f"{MODEL_ID}_responses.csv"
        self.df.to_csv(output_filepath)

    def prompt_gpt35_once(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125", messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing prompt: {prompt}\nError: {e}")
            return "Error in response"  # Placeholder or handle appropriately

    # TODO: generate all GPT-3.5 responses separately so we don't generate them every time for each LlaMA.
    def get_all_gpt35_responses(self) -> None:
        # self.df.loc[] = [
        #     self.prompt_gpt4_once(prompt) for prompt in self.df.loc[GENERAL_PROMPT]
        # ]

        # final_path = self.data_path.with_name("prepared_gpt4_responses_" + self.data_path.name)
        # self.df.to_csv(final_path)
        pass


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
    inferencer = Inferencer(data_path=INPUT_FILEPATH)
    inferencer.populate_df_general_prompts()
    inferencer.get_all_llama_responses()
