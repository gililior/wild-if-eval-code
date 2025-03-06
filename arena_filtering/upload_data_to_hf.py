import os
from argparse import ArgumentParser
from datasets import Dataset
import json


def push_to_hub(json_path, name_in_hub):
    hf_token = os.environ.get("HF_TOKEN")
    with open(json_path, 'rt') as f:
        data = json.load(f)
    task_ids = list(data.keys())
    decomposition = [data[task_id] for task_id in task_ids]
    ds = Dataset.from_dict({"conversation_id": task_ids, "decomposition": decomposition})
    ds.push_to_hub(name_in_hub, token=hf_token, split="test")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--decomposition", help="path to the decomposition file")
    parser.add_argument("--name_in_hub", help="name of the dataset in the hub")
    args = parser.parse_args()
    push_to_hub(args.decomposition, args.name_in_hub)


