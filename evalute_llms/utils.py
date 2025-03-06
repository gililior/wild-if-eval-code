

from arena_filtering.heuristic_filtering import leave_only_first_request
from arena_filtering.constants import LMSYS_NAME_IN_HUB, LMSYS_SPLIT
from datasets import load_dataset


def load_data(name_in_hub):
    decomposition_ds = load_dataset(name_in_hub, split="test")
    orig_ds = load_dataset(LMSYS_NAME_IN_HUB, split=LMSYS_SPLIT)
    conversation_ids = set(decomposition_ds["conversation_id"])
    orig_ds_filtered = orig_ds.filter(lambda x: x['conversation_id'] in conversation_ids)
    orig_ds_cleaned = orig_ds_filtered.map(leave_only_first_request)
    orig_ds_cleaned = orig_ds_cleaned.rename_column("conversation", "task")

    # add decomposition to the dataset
    # Convert decomposition_ds into a dictionary for fast lookup
    decomposition_dict = {row["conversation_id"]: row for row in decomposition_ds}

    # Merge using map function
    def merge_examples(example):
        match = decomposition_dict.get(example["conversation_id"], {})  # Find matching row in decomposition
        return {**example, **match}  # Merge dictionaries

    merged_dataset = orig_ds_cleaned.map(merge_examples)
    return merged_dataset
