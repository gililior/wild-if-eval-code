from matplotlib import pyplot as plt

from scripts.data_analysis.plots_for_paper import SCORES_DIR
import json
import numpy as np
import os

PREFIX = "llm-aaj-{judge}.{gen_model}-0shot-wild-if-eval.json"
PATH = os.path.join(SCORES_DIR, "other_judges", PREFIX)
JUDGE_MODELS = ["deepseek-v3", "llama3.3-70b", "qwen2.5-72b"]
GPT_MODEL_NAME = "gpt-4o-2024-08-06"
MODELS_TO_EVAL = ["llama3.3-70b", "mistral-large", "qwen2.5-72b", "llama3.1-405b", "deepseek-v3"]


if __name__ == '__main__':
    correlations_matrix = np.zeros((len(JUDGE_MODELS), len(MODELS_TO_EVAL)))
    accuracies_matrix = np.zeros((len(JUDGE_MODELS), len(MODELS_TO_EVAL)))
    for j, eval_model in enumerate(MODELS_TO_EVAL):
        with open(PATH.format(judge=GPT_MODEL_NAME, gen_model=eval_model), "r") as f:
            gpt_data = json.load(f)
        tasks_sorted = sorted(list(gpt_data.keys()))
        all_scores = []
        binary_scores = []
        for task in tasks_sorted:
            constraints_sorted = sorted(list(gpt_data[task]["scores"].keys()))
            for constraint in constraints_sorted:
                if gpt_data[task]["scores"][constraint] == "ERR":
                    gpt_data[task]["scores"][constraint] = 0
                all_scores.append(gpt_data[task]["scores"][constraint])
                binary_scores.append(gpt_data[task]["scores"][constraint]>0.5)
        for i, judge_model in enumerate(JUDGE_MODELS):
            with open(PATH.format(judge=judge_model, gen_model=eval_model), "r") as f:
                judge_data = json.load(f)
            all_scores_for_judge = []
            binary_scores_for_judge = []
            for task in tasks_sorted:
                constraints_sorted = sorted(list(gpt_data[task]["scores"].keys()))
                for constraint in constraints_sorted:
                    if judge_data[task]["scores"][constraint] == "ERR":
                        judge_data[task]["scores"][constraint] = 0
                    all_scores_for_judge.append(judge_data[task]["scores"][constraint])
                    binary_scores_for_judge.append(judge_data[task]["scores"][constraint]>0.5)
            accuracies_matrix[i, j] = np.sum(np.array(binary_scores_for_judge) == np.array(binary_scores)) / len(binary_scores)
            correlations_matrix[i, j] = np.corrcoef(np.array(all_scores_for_judge, dtype=np.float32), np.array(all_scores, dtype=np.float32))[0][1]
    plt.imshow(correlations_matrix, cmap='hot', vmin=0.5, vmax=1)
    plt.xticks(np.arange(len(MODELS_TO_EVAL)), MODELS_TO_EVAL, rotation=45)
    plt.yticks(np.arange(len(JUDGE_MODELS)), JUDGE_MODELS)
    plt.colorbar()
    plt.xlabel("Generation Model")
    plt.ylabel("Judge Model")
    plt.title("Correlation between Judge and GPT-4 scores")
    plt.tight_layout()
    plt.show()

    plt.imshow(accuracies_matrix, cmap='hot', vmin=0.8, vmax=1)
    plt.xticks(np.arange(len(MODELS_TO_EVAL)), MODELS_TO_EVAL, rotation=45)
    plt.yticks(np.arange(len(JUDGE_MODELS)), JUDGE_MODELS)
    plt.colorbar()
    plt.xlabel("Generation Model")
    plt.ylabel("Judge Model")
    plt.title("Accuracy of Judge model compared to GPT-4")
    plt.tight_layout()
    plt.show()

    mean_accuracies = np.mean(accuracies_matrix, axis=1).round(3)
    for i, judge_model in enumerate(JUDGE_MODELS):
        print(f"{judge_model}: {mean_accuracies[i]}")
