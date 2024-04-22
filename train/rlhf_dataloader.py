import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pandas import Index
from torch.utils.data import Dataset

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.helpers import get_openai_api_key  # noqa: E402

# df column names
QUESTION = "question"
RESOLUTION_CRITERIA = "resolution_criteria"
CHOSEN_PROMPTS = "CHOSEN_PROMPTS"
REJECTED_PROMPTS = "REJECTED_RESPONSES"
CHOSEN_RESPONSES = "CHOSEN_RESPONSES"
REJECTED_RESPONSES = "REJECTED_RESPONSES"


class ForecastingRLHF(Dataset):
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        assert "chosen" not in self._get_df_columns() and "rejected" not in self._get_df_columns()
        self.instruction_template = (
            "Below is an instruction that describes a task. "
            + "Write a response that appropriately completes the request.\n\n"
            + "### Instruction:\n{instruction}\n\n### Response:"
        )
        self.prompt_template = """Here is a forecasting question: {question}\n\nHere are the resolution criteria of the question: {resolution_criteria}.\n\nBased on the forecasting question and resolution criteria, generate {quality_type} quality step-by-step reasoning to create a resolution. Output exactly 6 steps without an introduction or a conclusion."""  # noqa: E501
        self.client = OpenAI(api_key=get_openai_api_key())

    def __len__(self) -> int:
        return len(self.df)

    def _get_df_columns(self) -> Index:
        return self.df.columns

    def format_prompt(self, question: str, resolution_criteria: str, is_high_quality: bool) -> str:
        assert isinstance(
            is_high_quality, bool
        ), "Please provide a boolean for the LLM prompt to generate high (if True) or low quality reasoning."  # noqa: E501
        assert len(question) > 0, f"Please provide a non-empty string for the {QUESTION}."
        assert len(resolution_criteria) > 0, f"Please provide a non-empty string for the {RESOLUTION_CRITERIA}."
        return self.prompt_template.format(
            question=question,
            resolution_criteria=resolution_criteria,
            quality_type="high" if is_high_quality else "low",
        )

    def populate_df_chosen_rejected(self) -> None:
        """
        Populate CHOSEN_PROMPTS and REJECTED_PROPMTS with the corresponding formatted prompts for each row in SELF.DF.
        """
        assert not (CHOSEN_PROMPTS in self._get_df_columns() or REJECTED_PROMPTS in self._get_df_columns())
        self.df[CHOSEN_PROMPTS] = self.df.apply(
            lambda row: self.format_prompt(row[QUESTION], row[RESOLUTION_CRITERIA], True), axis=1
        )
        self.df[REJECTED_PROMPTS] = self.df.apply(
            lambda row: self.format_prompt(row[QUESTION], row[RESOLUTION_CRITERIA], False), axis=1
        )

    def prompt_gpt4_once(self, prompt: str) -> str:
        assert len(prompt) > 0
        # TODO: error handling?
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-2024-04-09", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def get_all_gpt4_responses(self) -> None:
        """Call this method after `populate_df_chosen_rejected`."""
        assert CHOSEN_PROMPTS in self._get_df_columns() or REJECTED_PROMPTS in self._get_df_columns()
        assert not (
            CHOSEN_PROMPTS in self._get_df_columns() or REJECTED_PROMPTS in self._get_df_columns()
        ), f"Remove both columns {CHOSEN_PROMPTS} and {REJECTED_PROMPTS} from the df so they won't be overriden."
        chosen_responses = [self.prompt_gpt4_once(prompt) for prompt in self.df[CHOSEN_PROMPTS]]
        rejected_responses = [self.prompt_gpt4_once(prompt) for prompt in self.df[REJECTED_PROMPTS]]
        self.df[CHOSEN_RESPONSES] = chosen_responses
        self.df[REJECTED_RESPONSES] = rejected_responses

    def get_dataset(self) -> list[dict]:
        """TODO: Under construction. DO NOT use for now."""
        chosen = self.df[CHOSEN_RESPONSES].tolist()
        rejected = self.df[REJECTED_RESPONSES].tolist()
        prompts = [
            self.format_prompt(self.df["question"][idx], self.df["resolution_criteria"][idx])
            for idx in range(len(self.df))
        ]
        dataset = [{"prompt": prompts[i], "chosen": chosen[i], "rejected": rejected[i]} for i in range(len(prompts))]
        return dataset
