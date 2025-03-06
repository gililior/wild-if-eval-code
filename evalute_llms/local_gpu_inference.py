import os.path
from argparse import ArgumentParser
from evalute_llms.base_infer import InferenceBase
from evalute_llms.utils import load_data


class InitialResponse(InferenceBase):

    def load_data(self, data_path):
        return load_data(data_path)

    def get_key_in_out_dict(self):
        return "initial_responses"

    def get_data_for_inference(self):
        to_predict = []
        ordered_prompts = []
        for prompt in self.data["task"]:
            to_predict.append(
                [
                    {"role": "user", "content": prompt}
                ]
            )
            ordered_prompts.append(prompt)
        return to_predict, ordered_prompts

    def get_out_dict_format(self):
        out_dict = {
            "generator_model": self.model_name,
            "data_path": self.data_path,
        }
        return out_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="name or path to dataset to generate predictions for")
    parser.add_argument("--model", required=True,
                        help="path to model to generate predictions with")
    parser.add_argument("--out_path", required=True,
                        help="path to json file to save predictions to")

    args = parser.parse_args()
    inference_model = InitialResponse(args.model, args.dataset)
    inference_model.predict(args.out_path)
