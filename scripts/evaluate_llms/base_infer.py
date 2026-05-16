
import json
from vllm import LLM, SamplingParams
import os


class InferenceBase:
    def __init__(self, model, data_path):
        self.model_name = model
        self.sampling_params = SamplingParams(temperature=0, max_tokens=1000)
        self.data_path = data_path
        self.data = self.load_data(self.data_path)
        self.inference_model = self.load_model()

    def get_key_in_out_dict(self):
        raise NotImplementedError

    def load_data(self, data_path):
        raise NotImplementedError

    def load_model(self):
        print("LOADING INFERENCE MODEL...")
        if 'phi' in self.model_name.lower():
            llm = LLM(model=self.model_name, tokenizer=self.model_name, trust_remote_code=True)
        else:
            llm = LLM(model=self.model_name, tokenizer=self.model_name)
        return llm

    def get_data_for_inference(self):
        raise NotImplementedError

    def get_out_dict_format(self):
        raise NotImplementedError

    def predict(self, out_path):
        to_predict, ordered_prompts = self.get_data_for_inference()
        pred_with_outputs = self.get_predictions(to_predict, ordered_prompts)
        self.dump_output(ordered_prompts, out_path, pred_with_outputs)

    def dump_output(self, ordered_prompts, out_path, to_predict):
        out_dict = self.get_out_dict_format()
        predictions_key = self.get_key_in_out_dict()
        out_dict["predictions_key"] = predictions_key
        out_dict[predictions_key] = {}
        for i, prompt in enumerate(ordered_prompts):
            out_dict[predictions_key][prompt] = to_predict[i]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wt') as f:
            str_json = json.dumps(out_dict, indent=2)
            f.write(str_json)
        print(f"saved predictions to {out_path}")

    def get_predictions(self, to_predict, ordered_prompts):
        print("processing prompts...")
        print("generating responses...")
        outputs = self.inference_model.chat(messages=to_predict, sampling_params=self.sampling_params, use_tqdm=True)
        for i, prompt in enumerate(ordered_prompts):
            response = outputs[i].outputs[0].text
            to_predict[i].append({"role": "assistant", "content": response})
        return to_predict
