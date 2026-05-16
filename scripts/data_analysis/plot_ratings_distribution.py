import json
import numpy as np
from argparse import ArgumentParser


def main(path):
    with open(path, 'rt') as f:
        data = json.load(f)
    pos_scores = [data[prompt]['pos_score'] for prompt in data if data[prompt]['pos_score'] != 'ERR']
    pos_scores.sort()
    print("Distribution of pos_scores:")
    print(f"Min: {min(pos_scores)}")
    print(f"Max: {max(pos_scores)}")
    print(f"Mean: {np.mean(pos_scores)}")
    print(f"Median: {np.median(pos_scores)}")
    print(f"Standard deviation: {np.std(pos_scores)}")
    print(f"90% percentile: {np.percentile(pos_scores, 90)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scores", required=True,
                        help="path to JSON from classify_constrained_generation_tasks")
    args = parser.parse_args()
    main(args.scores)
