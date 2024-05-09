from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from datasets import Dataset as HF_Dataset


# df column names
QUESTION = "question"
RESOLUTION_CRITERIA = "resolution_criteria"
GENERAL_PROMPT = "general_prompt"
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

    def get_dataset(self) -> list[dict]:
        """Returns the dataset with each general prompt, chosen responses, and rejected responses."""
        for column in [GENERAL_PROMPT, CHOSEN_RESPONSE, REJECTED_RESPONSE]:
            assert self.df[column].apply(lambda x: isinstance(x, str) and len(x) > 0).all()
        dataset = {
            "prompt": self.df[GENERAL_PROMPT].tolist(),
            "chosen": self.df[CHOSEN_RESPONSE].tolist(),
            "rejected": self.df[REJECTED_RESPONSE].tolist()
        }
        return HF_Dataset.from_dict(dataset)
