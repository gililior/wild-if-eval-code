import json
from arena_filtering.classify_constrained_generation_tasks import ConstrainedGenerationClassification, BaseDataset
from arena_filtering.constants import DECOMPOSE_PROMPT
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


class DecomposerRITS(ConstrainedGenerationClassification):
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
    parser.add_argument("--positive_tasks", help="path to json with list of positive tasks")
    parser.add_argument("--out")
    parser.add_argument("--decompose_model")
    parser.add_argument("--API_key_name", default="RITS_API_KEY")
    parser.add_argument("--API_endpoint",
                        default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{}/v1")

    args = parser.parse_args()
    dataset = BaseDataset(args.positive_tasks)
    decomposer = DecomposerRITS(dataset, args.decompose_model, args.API_endpoint, args.API_key_name,
                                max_new_tokens=1000)
    decomposer.infer(args.out)


