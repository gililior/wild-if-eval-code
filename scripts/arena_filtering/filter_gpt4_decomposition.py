"""Reconciles model predictions across two generation runs so that every model
covers the same task set as the published `wild-if-eval-decomposed-gpt4` split.

This is a one-off plumbing script kept for paper-replication transparency. It
expects local directories of per-model prediction JSONs in the layout used
during the paper's compute runs; pass them via --first_dir / --second_dir."""

from argparse import ArgumentParser
import json
from datasets import load_dataset
import os
import random

random.seed(42)


def compare_missing_predictions(ds_filtered, first_dir, second_dir, out_dir):
    first_prefix = os.path.join(first_dir, "init-generations-via-rits-{model_name}.constrained-lmsys-chat-1m.json")
    second_prefix = os.path.join(second_dir, "init-generations-via-rits-{model_name}.temp-ds.json")
    missing_tasks = {}
    for model in ["deepseek-v3", "llama3.1-405b", "llama3.3-70b", "qwen2.5-72b", "mistral-large"]:
        missing_tasks[model] = set()
        with open(first_prefix.format(model_name=model), "r") as f:
            inference = json.load(f)
        with open(second_prefix.format(model_name=model), "r") as f:
            original = json.load(f)
        combined_data = {**inference, **original}
        new_data = {}
        for task in ds_filtered["task"]:
            if task not in combined_data:
                missing_tasks[model].add(task)
            else:
                new_data[task] = combined_data[task]
        print(len(new_data))
        with open(os.path.join(out_dir, f"{model}-0shot-wild-if-eval.json"), "w") as f:
            json.dump(new_data, f, indent=2)

    # check that the tasks are the same
    lengths = [len(tasks) for tasks in missing_tasks.values()]
    if len(set(lengths)) != 1:
        print("Different number of missing tasks for different models")
        return None
    else:
        print(f"{lengths[0]} missing tasks for all models")
        ds_filtered = ds_filtered.filter(lambda x: x["task"] not in missing_tasks[model])
        return ds_filtered


def combine_data_to_match_new_ds(ds_filtered, train_dir, test_dir, out_dir):
    first_dir = os.path.join(train_dir, "{model_name}-{it}-train-init-gen.json")
    second_dir = os.path.join(test_dir, "{model_name}-{it}-test-init-gen.json")
    missing_tasks = {}
    for model in ["gemma-2-2b", "gemma-2-9b", "Llama-3.1-8B", "Llama-3.2-1B", "Llama-3.2-3B", "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]:
        if "gemma" in model:
            it = "it"
        else:
            it = "Instruct"
        first_path = os.path.join(first_dir.format(model_name=model, it=it))
        with open(first_path, "r") as f:
            first_data = json.load(f)
        if "predictions_key" in first_data:
            first_data = first_data[first_data["predictions_key"]]
        second_path = os.path.join(second_dir.format(model_name=model, it=it))
        with open(second_path, "r") as f:
            second_data = json.load(f)
        if "predictions_key" in second_data:
            second_data = second_data[second_data["predictions_key"]]
        combined_data = {**first_data, **second_data}
        new_data = {}
        missing_tasks[model] = set()
        for task in ds_filtered["task"]:
            if task not in combined_data:
                print(f"Task {task} not found in {model}")
                missing_tasks[model].add(task)
            else:
                new_data[task] = combined_data[task]
        print(len(new_data))
        with open(os.path.join(out_dir, f"{model}-0shot-wild-if-eval.json"), "w") as f:
            json.dump(new_data, f, indent=2)
    lengths = [len(tasks) for tasks in missing_tasks.values()]
    if len(set(lengths)) != 1:
        print("Different number of missing tasks for different models")
        return None
    else:
        print(f"{lengths[0]} missing tasks for all models")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--name_in_hub", default='gililior/wild-if-eval-decomposed-gpt4',
                        help="HF dataset whose tasks define the target task set")
    parser.add_argument("--big_first_dir", required=True,
                        help="dir with first-pass predictions for the big models")
    parser.add_argument("--big_second_dir", required=True,
                        help="dir with second-pass predictions for the big models")
    parser.add_argument("--small_train_dir", required=True,
                        help="dir with train-init-gen JSONs for the small models")
    parser.add_argument("--small_test_dir", required=True,
                        help="dir with test-init-gen JSONs for the small models")
    parser.add_argument("--out_dir", required=True,
                        help="output dir for reconciled per-model prediction files")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load_dataset(args.name_in_hub, split="test")
    filtered_data = compare_missing_predictions(data, args.big_first_dir, args.big_second_dir, args.out_dir)
    if filtered_data is not None:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            filtered_data.push_to_hub(args.name_in_hub, token=hf_token, split="test")
        else:
            print("HF_TOKEN not set; skipping push to hub.")
    print(filtered_data)
    combine_data_to_match_new_ds(filtered_data, args.small_train_dir, args.small_test_dir, args.out_dir)
