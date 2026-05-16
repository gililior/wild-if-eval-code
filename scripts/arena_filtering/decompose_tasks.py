import json
from scripts.arena_filtering.classify_constrained_generation_tasks import ConstrainedGenerationClassification, BaseDataset
from scripts.arena_filtering.constants import DECOMPOSE_PROMPT
import re
from argparse import ArgumentParser


def filter_answer(text):
    if "Translated Constraints:" not in text:
        return []
    index = text.find("Translated Constraints:") + len("Translated Constraints:")
    answer = text[index:].strip()
    list_items = re.split(r'(?=\n\d+\.\s*[A-Z])', answer)

    # Remove empty strings from the list
    list_items = [item.strip() for item in list_items if item]

    # Remove numbers from the list items
    list_items = [re.sub(r'^\d+\.', '', item).strip() for item in list_items]
    if "" in list_items:
        print(answer)
    return list_items


class Decomposer(ConstrainedGenerationClassification):
    def _infer(self, task):
        message = DECOMPOSE_PROMPT.format(instruction=task)
        answer = self.get_answer(message)
        generated_text = answer["results"][0]["generated_text"]
        processed_answer = filter_answer(generated_text)
        return processed_answer

    def get_name(self):
        return "decomposition"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--positive_tasks", required=True,
                        help="path to JSON with the filtered list of positive tasks")
    parser.add_argument("--out", required=True)
    parser.add_argument("--decompose_model", required=True,
                        help="model name as exposed on the inference endpoint")
    parser.add_argument("--API_key_name", default="OPENAI_API_KEY",
                        help="name of env var holding the API key")
    parser.add_argument("--API_endpoint", default="https://api.openai.com/v1",
                        help="OpenAI-compatible chat completions endpoint")

    args = parser.parse_args()
    dataset = BaseDataset(args.positive_tasks)
    decomposer = Decomposer(dataset, args.decompose_model, args.API_endpoint, args.API_key_name,
                            max_new_tokens=1000)
    decomposer.infer(args.out)


