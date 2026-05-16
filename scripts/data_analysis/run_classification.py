"""Label constraints / tasks with an LLM: constraint generality, task domain,
or constraint category (single- or multi-label).

Runs against any OpenAI-compatible chat completions endpoint. The precomputed
classifications used by the paper figures are committed under `data/`
(`constraint_categories.csv`, `domains.csv`); this script is what produced them.

Example:
    export OPENAI_API_KEY=sk-...
    python -m scripts.data_analysis.run_classification \\
      --data_path gililior/wild-if-eval \\
      --split test \\
      --tasks_key task \\
      --classification_type domain \\
      --model gpt-4o-2024-08-06 \\
      --out_dir classification_out/
"""
import os
from argparse import ArgumentParser
import json
from multiprocessing import Pool, cpu_count
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset


class ClassificationData:
    def __init__(self, data_path, split, tasks_key):
        self.data = load_dataset(data_path, split=split)
        self.tasks_key = tasks_key

    def get_tasks_list(self):
        return list(self.data[self.tasks_key])

    def get_constraints_list(self):
        all_constraints = [c for sublist in self.data["decomposition"] for c in sublist]
        return list(set(all_constraints))


class Classifier:
    prompt_template = None  # set by subclasses

    def __init__(self, model, data, api_endpoint, api_key_name):
        self.model_name = model
        self.model_name_for_endpoint = model
        self.data = data
        self.api_endpoint = api_endpoint
        self.api_key = os.environ.get(api_key_name)

    def get_name(self):
        raise NotImplementedError

    def get_out_path(self, out_dir):
        return os.path.join(out_dir, f"{self.get_name()}-{self.model_name}.json")

    def dump_results(self, out_dir, all_results):
        os.makedirs(out_dir, exist_ok=True)
        with open(self.get_out_path(out_dir), 'wt') as f:
            f.write(json.dumps(all_results, indent=2))


class GeneralityClassification(Classifier):
    prompt_template =  \
        ("You are given a constraint from a generation task. "
         "Classify the generality of the constraint on a scale from 1 to 5, where 1 is the most general and 5 is the most specific. "
         "Provide your score using the format of [[rating]], for example: '[[3]]'. "
         "General constraints are constraints that can be combined with almost any generation task. "
         "In contrast, specific constraints can only be applied to quite particular situations and requests. "
         "Examples:\n"
         '- Constraint: "Keep the text short and concise." Score: [[1]] Explanation: This constraint is very general and can be added to almost any user request.\n'
         '- Constraint: "The target audience is non-financially aware non-reader young adults." Score: [[3]] Explanation: This is somewhat specific, but can still apply to different types of user requests.\n'
         '- Constraint: "Mention the company "Coca Cola"." Score: [[2]] Explanation: This constraint can in principle be added to a wide array of generative tasks.\n'
         '- Constraint: "Never come across as sounding redundant or repeating yourself." Score: [[1]] Explanation: This is a general guideline to the AI and is not task-specific.\n'
         '''- Constraint: "Describe the main character's desire for independence and his perception of himself as his own man." Score: [[3]] Explanation: This is a constraint that is only relevant for stories, but can apply to many story generation tasks.\n'''
         '- Constraint: "The hypothesis should be brand-new and not previously proposed." Score: [[4]] Explanation: This constraint will only be applicable to tasks where the assistant is asked to generate a hypothesis.\n'
         '- Constraint: "Explore the possibility of natural hybridization within the genus Sinocyclocheilus." Score: [[5]] Explanation: This is a very specific guideline that appears tied to a particular task.\n'
         '- Constraint: "The output should be in a well-structured JSON format with well-named keys." Score: [[2]] Explanation: The guideline is rather general, but not all tasks can adhere to this desired output format.\n'
         "\nConstraint: {}\n\nScore: ")

    def get_name(self):
        return "constraint-generality-classification"


class DomainClassification(Classifier):
    domains = "1. Creative Writing\n2. Chemical Industry\n3. Education\n4. Business\n5. Technology\n6. Healthcare\n7. Marketing\n8. Entertainment\n9. Environmental Science\n10. Psychology\n11. Roleplaying\n12. Science Fiction\n13. Fantasy\n14. Journalism\n15. Law\n16. Finance\n17. Data Analysis\n18. Artificial Intelligence\n19. Language Translation\n20. Gaming"

    prompt_template =  \
        ("You are given a generation task. "
         "Classify the domain of the task into one of the domains listed below. Respond only with the category number."
         "\nDomains:\n"+domains+"\n\nTask: {}\n\nYour response:")

    def get_name(self):
        return "task-domain-classification"


class ConstraintClassification(Classifier):
    def __init__(self, model, data, categories, api_endpoint, api_key_name):
        super().__init__(model, data, api_endpoint, api_key_name)

        self.categories_str = ""
        for j, category_dict in enumerate(categories):
            self.categories_str += f"\n{j}. *{category_dict['name']}*: {category_dict['description']}\nExamples: "
            for example in category_dict['examples']:
                self.categories_str += f"\n - {example}"

        self.prompt_template = (
                "Classify the following constraint from a generation task into one of the categories listed below. "
                "Respond only with the category number. "
                "If the constraint does not fit any of the categories from the list, respond with 'Other:' followed by a suggested title for an appropriate category.\n"
                "Categories:"+self.categories_str+"\n\nConstraint: {}\n\nYour response:")

    def get_name(self):
        return "constraint-classification-single"


class ConstraintMultilabelClassification(ConstraintClassification):
    def __init__(self, model, data, categories, api_endpoint, api_key_name):
        super().__init__(model, data, categories, api_endpoint, api_key_name)

        self.prompt_template = (
                "Classify the following constraint from a generation task into one (or more) of the categories listed below. "
                "Respond only with the category number(s). "
                "If the constraint fits multiple categories, provide the numbers separated by commas (e.g., '1,3,5'). "
                "If the constraint does not fit any of the categories from the list, respond with 'Other:' followed by a suggested title for an appropriate category.\n"
                "Categories:" + self.categories_str + "\n\nConstraint: {}\n\nYour response:")

    def get_name(self):
        return "constraint-classification-multilabel"


def generate_parallel(obj, constraints):
    model_name = obj.model_name
    api_key = obj.api_key
    base_url = obj.api_endpoint.format(obj.model_name_for_endpoint)
    prompt_template = obj.prompt_template
    all_results = {}
    all_args = {}
    pool = Pool(cpu_count())
    total = 0
    for task in constraints:
        all_args[task] = (task, api_key, base_url, model_name, prompt_template)
        total += 1
    pbar = tqdm(total=total)
    for task, arguments in all_args.items():
        all_results[task] = pool.apply_async(infer_local, arguments, callback=lambda _: pbar.update(1))
    pool.close()
    pool.join()
    print("DONE")
    return {task: task_result.get() for task, task_result in all_results.items()}


def infer_local(constraint, api_key, base_url, model_name, prompt_template):
    msg = prompt_template.format(constraint)
    message = [{'role': 'user', 'content': msg}]
    client = OpenAI(api_key=api_key, base_url=base_url)

    gen_params = {'temperature': 0}
    if client.base_url.host == "api.openai.com":
        gen_params["max_completion_tokens"] = 10
    else:
        gen_params['max_tokens'] = 10
        if "rits" in base_url:
            gen_params['extra_headers'] = {"RITS_API_KEY": api_key}

    completion = client.chat.completions.create(
        messages=message,
        model=model_name,
        **gen_params
    )
    generated_text = completion.choices[0].message.content
    return generated_text


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="HF dataset name or path")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", type=str, required=True,
                        help="model name as exposed on the inference endpoint")
    parser.add_argument("--tasks_key", required=True, help="the tasks column name")
    parser.add_argument("--API_key_name", default="OPENAI_API_KEY",
                        help="name of env var holding the API key")
    parser.add_argument("--API_endpoint", default="https://api.openai.com/v1",
                        help="OpenAI-compatible chat completions endpoint")
    parser.add_argument("--tasks_batch_size", type=int, default=200, help="number of tasks to run inference on before saving")
    parser.add_argument("--classification_type", type=str, required=True, choices=['generality', 'domain', "constraints_single", "constraints_multilabel"])
    parser.add_argument("--categories_file", type=str)

    args = parser.parse_args()

    dataset = ClassificationData(args.data_path, args.split, args.tasks_key)

    if args.classification_type == "domain":
        generator = DomainClassification(args.model, dataset, args.API_endpoint, args.API_key_name)
    elif args.classification_type == "generality":
        generator = GeneralityClassification(args.model, dataset, args.API_endpoint, args.API_key_name)
    else:
        with open(args.categories_file) as f:
            categories = json.load(f)

        if args.classification_type == "constraints_single":
            generator = ConstraintClassification(args.model, dataset, categories,
                                                 args.API_endpoint, args.API_key_name)
        elif args.classification_type == "constraints_multilabel":
            generator = ConstraintMultilabelClassification(args.model, dataset, categories,
                                                           args.API_endpoint, args.API_key_name)

    out_path = generator.get_out_path(args.out_dir)

    tasks_or_constraints = generator.data.get_tasks_list() if args.classification_type == "domain" \
        else generator.data.get_constraints_list()

    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        constraints = [con for con in set(tasks_or_constraints) if con not in existing]
        print(f"{len(existing)} already in file, {len(constraints)} to go")
    else:
        existing = {}
        constraints = list(set(tasks_or_constraints))

    all_generated = {}
    for i in range(0, len(constraints), args.tasks_batch_size):
        batch = constraints[i: i + args.tasks_batch_size]
        batch_generated = generate_parallel(generator, batch)
        all_generated = {**all_generated, **batch_generated}
        all_results_dict = {**existing, **all_generated}
        generator.dump_results(args.out_dir, all_results_dict)
