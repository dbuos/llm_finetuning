from datasets import load_dataset
import os
import pandas as pd


SRC_DIRNAME = os.path.join("data", "recogs")


def load_split(filename):
    return pd.read_csv(
        filename,
        delimiter="\t",
        names=['input', 'output', 'category'])

if __name__ == '__main__':
    recogs = load_split(f"{SRC_DIRNAME}/train.tsv")
    recogs = recogs.rename(columns={'input': 'question', 'output': 'answer'})
    recogs = recogs.drop(columns=['category'])
    save_dir = "raw_datasets/recogs_qa/"
    os.makedirs(save_dir, exist_ok=True)
    recogs.to_json('raw_datasets/recogs_qa/qa_recogs.json', orient='records', indent=4, lines=False)


