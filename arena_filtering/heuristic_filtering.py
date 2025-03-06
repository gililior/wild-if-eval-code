import os
from datasets import load_dataset
from argparse import ArgumentParser
import detoxify
from tqdm import tqdm
import json
from arena_filtering.constants import LMSYS_NAME_IN_HUB, LMSYS_SPLIT


def leave_only_first_request(example):
    example["conversation"] = example["conversation"][0]["content"]
    return example


def filter_data(dataset):
    updated_dataset = dataset.map(leave_only_first_request)
    rename_col = updated_dataset.rename_column("conversation", "task")

    no_code_only_english = rename_col.filter(lambda example:
                                             "code" not in example["task"].lower() and example["language"] == "English")

    # filter out all toxic samples
    classifier = detoxify.Detoxify("original", device='mps')
    to_remove = []
    for text in tqdm(no_code_only_english['task'], desc="Processing"):
        result = classifier.predict(text)
        if result['toxicity'] > 0.6:
            to_remove.append(text)
    non_toxic = no_code_only_english.filter(lambda x: x["task"] not in to_remove)

    seen = set()

    def remove_duplicates(example):
        key = tuple(example[col] for col in ["task"])
        if key in seen:
            return False
        seen.add(key)
        return True

    dedup = non_toxic.filter(remove_duplicates)

    ids = [example["conversation_id"] for example in dedup]

    return ids


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--out_path", help="path to save the filtered ids")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    ds = load_dataset(LMSYS_NAME_IN_HUB, split=LMSYS_SPLIT)

    filtered_ids = filter_data(ds)
    with open(args.out_path, 'wt') as f:
        json.dump(filtered_ids, f)
    print(f"saved at {args.out_path}")

