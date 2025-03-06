
from argparse import ArgumentParser
from datasets import load_from_disk
from arena_filtering.constants import FILTER_PROMPT
import os
import re
import numpy as np
import concurrent.futures
from tqdm import tqdm
import json
from openai import OpenAI

from datasets import load_dataset
from constants import LMSYS_NAME_IN_HUB, LMSYS_SPLIT, ERROR_SCORE
from arena_filtering.heuristic_filtering import leave_only_first_request


class BaseDataset:

    def __init__(self, path_to_ids):
        self.lmsys_data = load_dataset(LMSYS_NAME_IN_HUB, split=LMSYS_SPLIT)
        self.filtered_ids = []
        with open(path_to_ids, 'rt') as f:
            self.filtered_ids = json.load(f)
        self.data = self.lmsys_data.filter(lambda x: x['conversation_id'] in self.filtered_ids)
        self.data.rename_column("conversation", "task")
        self.data["task"] = self.data["task"].map(leave_only_first_request)

    def get_tasks_list(self):
        return self.data["task"]

    def get_id(self, task):
        for i, t in enumerate(self.data["task"]):
            if t == task:
                return self.data["conversation_id"][i]
        else:
            return None


class ConstrainedGenerationClassification:
    MAX_WORKERS = 20

    def __init__(self, data, model_name, api_endpoint, api_key_name, max_new_tokens):
        self.model_name_for_generation = self.get_model_name_for_generation(model_name)
        self.api_endpoint = api_endpoint
        self.api_key_name = api_key_name
        self.model_name_for_endpoint = self.get_model_name_in_server(self.model_name_for_generation)
        self.client = OpenAI(api_key=os.environ.get(api_key_name),
                             base_url=api_endpoint.format(self.model_name_for_endpoint))
        self.data = data
        self.short_model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.generation_params = self.get_generation_params()

    def get_name(self):
        return f"constrained-gen-pos-score"

    def get_out_path(self, out_dir):
        path = os.path.join(out_dir, f"{self.get_name()}-{self.short_model_name}.json")
        print(f"output path at {path}")
        return path

    def infer(self, out_dir):
        answers = {}
        out_path = self.get_out_path(out_dir)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        future_to_task = {}

        if os.path.exists(out_path):
            with open(out_path, 'rt') as f:
                answers = json.load(f)

        feedback_file = open(out_path, 'wt')
        for task in set(self.data.get_tasks_list()):
            if task in answers:
                continue
            future_to_task[pool.submit(self._infer, task)] = task

        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(future_to_task)):
            task = future_to_task[future]
            feedback_dict = future.result()
            task_conversation_id = self.data.get_id(task)
            answers[task_conversation_id] = feedback_dict
            self.redump_json(feedback_file, answers)

        pool.shutdown(wait=True)
        self.redump_json(feedback_file, answers)
        feedback_file.close()

    @staticmethod
    def redump_json(feedback_file, answers):
        str_feedback_dict = json.dumps(answers, indent=2)
        feedback_file.seek(0)
        feedback_file.write(str_feedback_dict)

    def _infer(self, task):
        message = FILTER_PROMPT.format(request=task)
        answer = self.get_answer(message)
        generated_text = answer["results"][0]["generated_text"]
        generated_tokens = answer["results"][0]["generated_tokens"]
        pos_score = self.calc_score(generated_tokens)
        return {"answer": generated_text, "pos_score": pos_score}

    @staticmethod
    def get_model_name_for_generation(model_name):
        if model_name == 'llama3-70b':
            full_name = 'meta-llama/llama-3-70b-instruct'
        elif model_name == 'llama3.1-70b':
            full_name = 'meta-llama/llama-3-1-70b-instruct'
        elif model_name == 'llama3.3-70b':
            full_name = 'meta-llama/llama-3-3-70b-instruct'
        elif model_name == 'llama3-405b':
            full_name = 'meta-llama/llama-3-405b-instruct'
        elif model_name == 'llama3.1-405b':
            full_name = 'meta-llama/llama-3-1-405b-instruct-fp8'
        elif model_name == 'llama3.1-8b':
            full_name = 'meta-llama/Llama-3.1-8B-Instruct'
        elif model_name == 'qwen2.5-72b':
            full_name = 'Qwen/Qwen2.5-72B-Instruct'
        elif model_name == 'deepseek-v3':
            full_name = 'deepseek-ai/DeepSeek-V3'
        elif model_name == 'mistral-large':
            full_name = 'mistralai/mistral-large-instruct-2407'
        else:
            raise RuntimeError(f"model unknown {model_name}")
        return full_name

    def get_model_name_in_server(self, model_name_for_generation):
        if 'rits' in self.api_endpoint:
            full_name = model_name_for_generation.split(
                "/")[-1].lower().replace("v0.1", "v01").replace(".", "-")
        else:
            full_name = model_name_for_generation
        return full_name

    @staticmethod
    def calc_score(token_preds: list[dict]):
        num_tokens_to_check = 5
        min_probability_mass = 0.0001
        for i in range(min(num_tokens_to_check, len(token_preds))):
            try:
                pos_probs, neg_probs = ConstrainedGenerationClassification.get_pos_neg_probs(
                    token_logprobs_obj=token_preds[i]["top_tokens"])
                if pos_probs or neg_probs:
                    sum_probs = sum(pos_probs) + sum(neg_probs)
                    if sum_probs > min_probability_mass:
                        return sum(pos_probs) / sum_probs
            except:
                pass
        return ERROR_SCORE

    @staticmethod
    def get_pos_neg_probs(token_logprobs_obj):
        pos_and_neg_probs = []
        for class_name in ["yes", "no"]:
            # We need to capture different variants of model behavior and tokenizers, for example with opening space,
            # punctuation etc. but avoid longer words that contain the class name.
            # For example, for class "yes" we would capture "YES," and " Yes" but not "yesterday".
            name_regex = re.compile(
                rf"(\W|Ġ|_)*{class_name}(\W|Ġ|_)*", flags=re.IGNORECASE
            )
            class_probs = [
                np.exp(d["logprob"])
                for d in token_logprobs_obj
                if name_regex.fullmatch(d["text"])
            ]
            pos_and_neg_probs.append(class_probs)
        return pos_and_neg_probs

    def get_generation_params(self):
        gen_params = {'temperature': 0}
        if self.client.base_url.host == "api.openai.com":
            gen_params["max_completion_tokens"] = self.max_new_tokens
        else:
            gen_params['extra_headers'] = {"RITS_API_KEY": os.environ.get(self.api_key_name)}
            gen_params['max_tokens'] = self.max_new_tokens

        gen_params['top_logprobs'] = 5  # numbers of tokens to evaluate upon
        gen_params['logprobs'] = True
        return gen_params

    def get_answer(self, message):
        if type(message) is str:
            message = [{'role': 'user', 'content': message}]
        completion = self.client.chat.completions.create(
            messages=message,
            model=self.model_name_for_generation,
            **self.generation_params
        )
        answer = {'results': []}
        generated_text = completion.choices[0].message.content
        top_logprobs_response = completion.choices[0].logprobs.content
        token_dicts = [
            {
                "top_tokens": [
                    {"text": obj.token, "logprob": obj.logprob}
                    for obj in generated_token.top_logprobs
                ]
            }
            for generated_token in top_logprobs_response
        ]
        answer['results'].append({"generated_tokens": token_dicts, "generated_text": generated_text})
        return answer



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--path_to_filtered_ids")
    parser.add_argument("--out_dir")
    parser.add_argument("--classification_model")
    parser.add_argument("--API_key_name", default="RITS_API_KEY")
    parser.add_argument("--API_endpoint", default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{}/v1")
    args = parser.parse_args()
    dataset = BaseDataset(args.path_to_filtered_ids)
    classifier = ConstrainedGenerationClassification(dataset, args.model_name,
                                                     args.API_endpoint, args.API_key_name,
                                                     max_new_tokens=5)
    classifier.infer(args.out_dir)
