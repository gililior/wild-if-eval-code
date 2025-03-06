import os
from argparse import ArgumentParser
from evalute_llms.api_inference import NewData, InitGenerationsAPI
from evalute_llms.constants import PROMPT_EVAL
import random
import json
from multiprocessing import Pool, cpu_count
from openai import OpenAI
from tqdm import tqdm


class InferenceData(NewData):
    def __init__(self, ds_name, path_to_predictions, sample):
        super().__init__(ds_name)
        self.predictions = self.load_predictions(path_to_predictions)
        self.num_sample = sample

    def load_predictions(self, name_or_path):
        with open(name_or_path, 'rt') as f:
            data = json.load(f)
        pred_key = data.get("predictions_key")
        if pred_key is not None:
            data = data[pred_key]
        return data

    def get_response(self, task):
        return self.predictions[task][-1]["content"]

    def get_tasks_list(self):
        list_of_tasks = super().get_tasks_list()
        if self.num_sample > -1:
            sorted_tasks = sorted(list_of_tasks)
            list_of_tasks = random.Random(42).sample(sorted_tasks, k=self.num_sample)
        return list_of_tasks


class LLMJudgeConstraintsRITS(InitGenerationsAPI):

    def get_name(self):
        return f"decomposition-evaluation"


def generate_parallel(obj, tasks):
    model_name = obj.model_name_for_generation
    api_key = os.environ.get(obj.api_key_name)
    base_url = obj.api_endpoint.format(obj.model_name_for_endpoint)
    all_results = {}
    all_args = {}
    pool = Pool(cpu_count())
    total = 0
    for task in tasks:
        all_args[task] = {}
        response = obj.data.get_response(task)
        for atomic in obj.data.get_constraints(task):
            all_args[task][atomic] = (task, response, atomic)
            total += 1
    pbar = tqdm(total=total)
    for task in all_args:
        all_results[task] = {}
        for atomic in all_args[task]:
            arguments = all_args[task][atomic] + (api_key, base_url, model_name)
            all_results[task][atomic] = pool.apply_async(infer_local, arguments, callback=lambda _: pbar.update(1))
    pool.close()
    pool.join()
    print("DONE")
    return all_results


def process_results(all_results):
    processed_results = {}

    for task in all_results:
        processed_results[task] = {"scores": {}, "explanations": {}}
        for atomic in all_results[task]:
            result = all_results[task][atomic].get()
            processed_results[task]["scores"][atomic] = result[0]
            processed_results[task]["explanations"][atomic] = result[1]
    return processed_results


def infer_local(task, response, atomic, api_key, base_url, model_name):
    message = PROMPT_EVAL.format(instruction=task, response=response, constraint=atomic) 
    message = [{'role': 'user', 'content': message}]
    client = OpenAI(api_key=api_key, base_url=base_url)
    max_new_tokens = 1000
    gen_params = {'temperature': 0}
    if client.base_url.host == "api.openai.com":
        gen_params["max_completion_tokens"] = max_new_tokens
    else:
        gen_params['extra_headers'] = {"RITS_API_KEY": api_key}
        gen_params['max_tokens'] = max_new_tokens

    gen_params['top_logprobs'] = 5  # numbers of tokens to evaluate upon
    gen_params['logprobs'] = True

    completion = client.chat.completions.create(
            messages=message,
            model=model_name,
            **gen_params
        )
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

    pos_score = LLMJudgeConstraintsRITS.calc_score(token_dicts)
    results_for_atomic = pos_score
    explanations_for_atomic = {"tokens": token_dicts, "text": generated_text}

    return results_for_atomic, explanations_for_atomic


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--eval_model")
    parser.add_argument("--to_eval", help="path to the responses to evaluate")
    parser.add_argument("--API_key_name", default="RITS_API_KEY")
    parser.add_argument("--API_endpoint",
                        default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{}/v1")
    parser.add_argument("--data", help="ds name in hub")
    parser.add_argument("--sample", type=int, default=-1,
                        help="specify how many samples to evaluate")
    parser.add_argument("--tasks_batch_size", type=int, default=200,
                        help="number of tasks to run inference on before saving")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    print(f"\n\n=======\nEVALUATING {args.to_eval} WITH {args.eval_model}")
    dataset = InferenceData(args.data, args.to_eval, args.sample)

    classifier = LLMJudgeConstraintsRITS(dataset, args.eval_model, args.API_endpoint,
                                         args.API_key_name, max_new_tokens=5)
    out_path = classifier.get_out_path(args.out_dir)
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        all_tasks = [task for task in set(classifier.data.get_tasks_list()) if task not in existing]
        print(f"{len(existing)} already in file, {len(all_tasks)} to go")
    else:
        existing = {}
        all_tasks = list(set(classifier.data.get_tasks_list()))

    all_generated = {}
    for i in range(0, len(all_tasks), args.tasks_batch_size):
        batch = all_tasks[i: i + args.tasks_batch_size]
        batch_generated = generate_parallel(classifier, batch)
        all_generated = {**all_generated, **batch_generated}
        all_results_dict = {**existing, **process_results(all_generated)}
        classifier.dump_results(args.out_dir, all_results_dict)
