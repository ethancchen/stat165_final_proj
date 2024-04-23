from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

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
        dataset = self.df.apply(
            lambda row: {
                "prompt": row[GENERAL_PROMPT],
                "chosen": row[CHOSEN_RESPONSE],
                "rejected": row[REJECTED_RESPONSE],
            },
            axis=1,
        ).tolist()
        return dataset
