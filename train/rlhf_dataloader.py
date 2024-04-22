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
GENERAL_PROMPT = "general_prompt"
CHOSEN_PROMPT = "chosen_prompt"
REJECTED_PROMPT = "rejected_response"
CHOSEN_RESPONSE = "chosen_response"
REJECTED_RESPONSE = "rejected_response"


class ForecastingRLHF(Dataset):
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.instruction_template = (
            "Below is an instruction that describes a task. "
            + "Write a response that appropriately completes the request.\n\n"
            + "### Instruction:\n{instruction}\n\n### Response:"
        )
        self.general_prompt_template = """Here is a forecasting question: {question}\n\nHere are the resolution criteria of the question: {resolution_criteria}.\n\nBased on the forecasting question and resolution criteria, generate step-by-step reasoning to create a resolution. Output exactly 6 steps without an introduction or a conclusion."""  # noqa: E501
        self.specific_prompt_template = """Here is a forecasting question: {question}\n\nHere are the resolution criteria of the question: {resolution_criteria}.\n\nBased on the forecasting question and resolution criteria, generate {quality_type} quality step-by-step reasoning to create a resolution. Output exactly 6 steps without an introduction or a conclusion."""  # noqa: E501
        self.client = OpenAI(api_key=get_openai_api_key())

    def __len__(self) -> int:
        return len(self.df)

    def _get_df_columns(self) -> Index:
        return self.df.columns

    def format_specific_prompt(self, question: str, resolution_criteria: str, is_high_quality: bool) -> str:
        assert isinstance(
            is_high_quality, bool
        ), "Please provide a boolean for the LLM prompt to generate high (if True) or low quality reasoning."  # noqa: E501
        assert len(question) > 0, f"Please provide a non-empty string for the {QUESTION}."
        assert len(resolution_criteria) > 0, f"Please provide a non-empty string for the {RESOLUTION_CRITERIA}."
        return self.specific_prompt_template.format(
            question=question,
            resolution_criteria=resolution_criteria,
            quality_type="high" if is_high_quality else "low",
        )

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

    def populate_df_chosen_rejected(self) -> None:
        """
        Populate CHOSEN_PROMPT and REJECTED_PROPMT with the corresponding formatted prompts for each row in SELF.DF.
        """
        assert not (CHOSEN_PROMPT in self._get_df_columns() or REJECTED_PROMPT in self._get_df_columns())
        self.df[CHOSEN_PROMPT] = self.df.apply(
            lambda row: self.format_specific_prompt(row[QUESTION], row[RESOLUTION_CRITERIA], True), axis=1
        )
        self.df[REJECTED_PROMPT] = self.df.apply(
            lambda row: self.format_specific_prompt(row[QUESTION], row[RESOLUTION_CRITERIA], False), axis=1
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
        assert CHOSEN_PROMPT in self._get_df_columns() or REJECTED_PROMPT in self._get_df_columns()
        assert not (
            CHOSEN_PROMPT in self._get_df_columns() or REJECTED_PROMPT in self._get_df_columns()
        ), f"Remove both columns {CHOSEN_PROMPT} and {REJECTED_PROMPT} from the df so they won't be overriden."
        chosen_responses = [self.prompt_gpt4_once(prompt) for prompt in self.df[CHOSEN_PROMPT]]
        rejected_responses = [self.prompt_gpt4_once(prompt) for prompt in self.df[REJECTED_PROMPT]]
        self.df[CHOSEN_RESPONSE] = chosen_responses
        self.df[REJECTED_RESPONSE] = rejected_responses

    def get_dataset(self) -> list[dict]:
        """Returns the dataset with each general prompt, chosen responses, and rejected responses."""
        for column in [GENERAL_PROMPT, CHOSEN_RESPONSE, REJECTED_RESPONSE]:
            assert self.df[column].apply(lambda x: isinstance(x, str) and len(x) > 0).all()
        dataset = self.df.apply(
            lambda row: {
                "prompt": row[GENERAL_PROMPT],
                "chosen": row[CHOSEN_RESPONSE],
                "rejected": row[REJECTED_RESPONSE],
            },
            axis=1,
        ).tolist()
        return dataset
