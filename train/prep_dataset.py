import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pandas import Index

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from utils.helpers import get_openai_api_key  # noqa: E402

# df column names
QUESTION = "question"
RESOLUTION_CRITERIA = "resolution_criteria"
GENERAL_PROMPT = "general_prompt"
CHOSEN_PROMPT = "chosen_prompt"
REJECTED_PROMPT = "rejected_prompt"
CHOSEN_RESPONSE = "chosen_response"
REJECTED_RESPONSE = "rejected_response"


class PrepDataset:
    def __init__(self, data_path: Path) -> None:
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
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-2024-04-09", messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing prompt: {prompt}\nError: {e}")
            return "Error in response"  # Placeholder or handle appropriately

    def get_all_gpt4_responses(self) -> None:
        assert CHOSEN_PROMPT in self._get_df_columns() or REJECTED_PROMPT in self._get_df_columns()
        assert not (
            CHOSEN_RESPONSE in self._get_df_columns() or REJECTED_RESPONSE in self._get_df_columns()
        ), f"Remove both columns {CHOSEN_RESPONSE} and {REJECTED_RESPONSE} from the df so they won't be overridden."

        batch_size = 100
        total_rows = len(self.df)
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            print(f"Processing rows {start+1} to {end}...")

            self.df.loc[start:end, CHOSEN_RESPONSE] = [
                self.prompt_gpt4_once(prompt) for prompt in self.df.loc[start:end, CHOSEN_PROMPT]
            ]
            self.df.loc[start:end, REJECTED_RESPONSE] = [
                self.prompt_gpt4_once(prompt) for prompt in self.df.loc[start:end, REJECTED_PROMPT]
            ]

            # Save progress
            backup_path = self.data_path.with_name(f"prepared_gpt4_responses_checkpoint_{start}_{end}.csv")
            self.df.to_csv(backup_path)
            print(f"Checkpoint saved to {backup_path}")

        # Final save
        final_path = self.data_path.with_name("prepared_gpt4_responses_" + self.data_path.name)
        self.df.to_csv(final_path)
        print(f"Final data saved to {final_path}")
