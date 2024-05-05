from datetime import datetime
from pathlib import Path

import pandas as pd

CSV_DIR = Path().resolve().parent / "data"
DATE_RESOLVE_AT = "date_resolve_at"
QUESTION = "question"
RESOLUTION_CRITERIA = "resolution_criteria"
LLAMA_3_8B_KNOWLEDGE_CUTOFF = (
    "2023-03-31"  # March 2023 (no day specified online), SEE: https://huggingface.co/meta-llama/Meta-Llama-3-8B
)
NUM_SAMPLES = 100
INPUT_FILE_NAME = "test_hf_dataset.csv"
OUTPUT_FILE_NAME = "sampled_100_test_hf_dataset.csv"
INPUT_FILE_PATH = CSV_DIR / INPUT_FILE_NAME
OUTPUT_FILE_PATH = CSV_DIR / OUTPUT_FILE_NAME


def prep_test_set() -> None:
    def parse_datetime(s: str) -> datetime:
        format = "%Y-%m-%d"
        return datetime.strptime(s, format)

    df = pd.read_csv(INPUT_FILE_PATH)
    df[DATE_RESOLVE_AT] = df[DATE_RESOLVE_AT].apply(parse_datetime)
    threshold = datetime.strptime(LLAMA_3_8B_KNOWLEDGE_CUTOFF, "%Y-%m-%d")
    assert df[DATE_RESOLVE_AT].min() > threshold
    df = df.sample(NUM_SAMPLES).filter(items=[QUESTION, RESOLUTION_CRITERIA])
    df.to_csv(OUTPUT_FILE_PATH, index=False)


if __name__ == "__main__":
    prep_test_set()
