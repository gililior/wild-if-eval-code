import os
from argparse import ArgumentParser
from arena_filtering.classify_constrained_generation_tasks import ConstrainedGenerationClassification, BaseDataset
import json
from multiprocessing import Pool, cpu_count
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset
from evalute_llms.utils import load_data


class NewData:

    def __init__(self, name_or_path):
        self.data = load_data(name_or_path)

    def get_tasks_list(self):
        return list(self.data["task"])

    def get_constraints_list(self):
        all_constraints = [item for sublist in self.data["decomposition"] for item in sublist]
        all_unique_constraints = list(set(all_constraints))
        return all_unique_constraints

    def get_constraints_for_task(self, task):
        task_idx = self.data["task"].index(task)
        return self.data["decomposition"][task_idx]


class InitGenerationsAPI(ConstrainedGenerationClassification):

    def get_name(self):
        return f"init-generations-via-rits"

    def get_out_path(self, out_dir):
        dataset_name = "wild-if-eval"
        path = os.path.join(out_dir, f"{self.get_name()}-{self.short_model_name}.{dataset_name}.json")
        print(f"output path at {path}")
        return path

    def dump_results(self, out_dir, all_scores):
        out_path_dump = self.get_out_path(out_dir)
        with open(out_path_dump, 'wt') as f:
            str_feedback_dict = json.dumps(all_scores, indent=2)
            f.write(str_feedback_dict)


def generate_parallel(obj, tasks):
    model_name = obj.model_name_for_generation
    api_key = os.environ.get(obj.api_key_name)
    base_url = obj.api_endpoint.format(obj.model_name_for_endpoint)
    all_results = {}
    all_args = {}
    pool = Pool(cpu_count())
    total = 0
    for task in tasks:
        all_args[task] = (task, api_key, base_url, model_name)
        total += 1
    pbar = tqdm(total=total)
    for task, arguments in all_args.items():
        all_results[task] = pool.apply_async(infer_local, arguments, callback=lambda _: pbar.update(1))
    pool.close()
    pool.join()
    print("DONE")
    return {task: task_result.get() for task, task_result in all_results.items()}


def infer_local(task, api_key, base_url, model_name):
    message = [{'role': 'user', 'content': task}]
    client = OpenAI(api_key=api_key, base_url=base_url)

    gen_params = {
        'max_new_tokens': 1000,
        'temperature': 0,
        'extra_headers': {"RITS_API_KEY": api_key}
    }

    if client.base_url.host == "api.openai.com":
        gen_params["max_completion_tokens"] = gen_params.pop("max_new_tokens", None)
    else:
        gen_params['max_tokens'] = gen_params.pop("max_new_tokens", None)

    completion = client.chat.completions.create(
        messages=message,
        model=model_name,
        **gen_params
    )
    generated_text = completion.choices[0].message.content
    return [{"role": "user", "content": task}, {"role": "assistant", "content": generated_text}]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--API_key_name", default="RITS_API_KEY")
    parser.add_argument("--API_endpoint",
                        default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{}/v1")
    parser.add_argument("--data", help="ds name in hub")
    parser.add_argument("--tasks_batch_size", type=int, default=200,
                        help="number of tasks to run inference on before saving")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    dataset = NewData(args.data_path)

    generator = InitGenerationsAPI(dataset, args.model_name, args.API_endpoint, args.API_key_name,
                                   max_new_tokens=1000)
    out_path = generator.get_out_path(args.out_dir)

    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        all_tasks = [task for task in set(generator.data.get_tasks_list()) if task not in existing]
        print(f"{len(existing)} already in file, {len(all_tasks)} to go")
    else:
        existing = {}
        all_tasks = list(set(generator.data.get_tasks_list()))

    all_generated = {}
    for i in range(0, len(all_tasks), args.tasks_batch_size):
        batch = all_tasks[i: i + args.tasks_batch_size]
        batch_generated = generate_parallel(generator, batch)
        all_generated = {**all_generated, **batch_generated}
        all_results_dict = {**existing, **all_generated}
        generator.dump_results(args.out_dir, all_results_dict)
