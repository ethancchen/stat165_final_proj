from pathlib import Path

import pandas as pd
from datasets import load_dataset

HF_DATA_PATH = "YuehHanChen/forecasting"
DATA_SPLITS = ("train", "validation")
CSV_DIR = Path(__file__).resolve().parents[1] / "data"


def download_initial_dataset():
    """
    Fetches the train and val HuggingFace datasets from Prof's paper and saves them as CSVs.
    Currently lets the HF_TOKEN be found implicitly.
    """
    for data_split in DATA_SPLITS:
        dataset = load_dataset(path=HF_DATA_PATH, split=data_split)
        df = pd.DataFrame(dataset)
        df.to_csv(CSV_DIR / f"{data_split}_hf_dataset.csv", index=False)


if __name__ == "__main__":
    CSV_DIR.mkdir(exist_ok=True)
    download_initial_dataset()
