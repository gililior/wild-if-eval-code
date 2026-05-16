
import json
import re
path_to_predictions = "model_predictions/{model}-0shot-wild-if-eval.json"
path_to_scores = "llm_aaj_scores/llm-aaj-deepseek-v3.{model}-0shot-wild-if-eval.json"

def main(path_scores, path_predictions):
    """
    Compare the length of the predictions and scores.
    """
    with open(path_predictions, "r") as predictions_file:
        predictions = json.load(predictions_file)
    with open(path_scores, "r") as scores_file:
        scores = json.load(scores_file)

    matches = []
    for task in scores:
        if task not in predictions:
            print(f"Task {task} is missing in predictions.")
            continue

        for constraint in scores[task]["scores"]:
            if re.search("\d+ words", constraint):  # Check if the constraint is about length in words
                length_max, length_min = -1, -1
                if "between" in constraint: # between
                    length_max = int(re.search("(\d+).*?(\d+)", constraint).group(1))
                    length_min = int(re.search("(\d+).*?(\d+)", constraint).group(2))
                else: # at most, up to, within, no more than, maximum
                    is_up_to = False
                    for word in ["within", "up to", "at most", "no more than", "less", "maximum", "under"]:
                        if word in constraint:
                            is_up_to = True
                            break
                    if not is_up_to:
                        continue
                    length_max = int(re.search("(\d+) words", constraint).group(1))
                    length_min = 0
                if length_min == length_max == -1:
                    continue
                heuristic_true = length_min <= len(predictions[task][-1]["content"].split(' ')) <= length_max
                model_true = scores[task]["scores"][constraint] >= 0.5
                matches.append(model_true == heuristic_true)

    print(f"Number of tasks with length in words constraint: {len(matches)}")
    print(f"Number of tasks with matching heuristic and model: {sum(matches)}")
    print(f"Percentage of matching tasks: {sum(matches) / len(matches) * 100:.2f}%")

if __name__ == "__main__":
    for model in ["deepseek-v3", "llama3.1-405b", "llama3.3-70b", "mistral-large"]:
        path_to_predictions_model = path_to_predictions.format(model=model)
        path_to_scores_model = path_to_scores.format(model=model)
        print(f"Comparing lengths for model: {model}")
        main(path_to_scores_model, path_to_predictions_model)

