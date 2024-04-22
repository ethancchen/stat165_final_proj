import pandas as pd
from torch.utils.data import Dataset


class ForecastingRLHF(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.instruction_template = (
            "Below is an instruction that describes a task. "
            + "Write a response that appropriately completes the request.\n\n"
            + "### Instruction:\n{instruction}\n\n### Response:"
        )
        self.prompt_template = """Here is a forecasting question: {question}\n\nHere are the resolution criteria of the question: {resolution_criteria}.\n\nBased on the forecasting question and resolution criteria, generate {quality_type} quality step-by-step reasoning to create a resolution. Output exactly 6 steps without an introduction or a conclusion."""  # noqa: E501

    def __len__(self):
        return len(self.df)

    def format_prompt(self, question: str, resolution_criteria: str) -> str:
        return self.instruction_template.format(
            instruction=self.prompt_template.format(question=question, resolution_criteria=resolution_criteria)
        )

    def get_dataset(self) -> list[dict]:
        chosen = self.df["chosen"].tolist()
        rejected = self.df["rejected"].tolist()
        prompts = [
            self.format_prompt(self.df["question"][idx], self.df["resolution_criteria"][idx])
            for idx in range(len(self.df))
        ]
        dataset = [{"prompt": prompts[i], "chosen": chosen[i], "rejected": rejected[i]} for i in range(len(prompts))]
        return dataset
