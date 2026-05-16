import os.path
from argparse import ArgumentParser
import json
import numpy as np
import matplotlib.pyplot as plt


def run(path_to_scores_file, out_dir, percentile, threshold):
    with open(path_to_scores_file, 'rt') as f:
        scores = json.load(f)

    if percentile > 0:
        # filter by percentile
        pos_scores = np.array([scores[sample]["pos_score"] for sample in scores
                               if scores[sample]["pos_score"] != "ERR"])
        threshold = np.percentile(pos_scores, percentile)
        out_path = os.path.join(out_dir, f"filtered_{round(percentile/100, 2)}percentile_{round(threshold,2)}threshold.json")
        print(f"{path_to_scores_file} {percentile} top pos scores requires a threshold of {threshold}")
    else:
        # filter by pre-defined threshold
        out_path = os.path.join(out_dir, f"filtered_{threshold}threshold.json")
    print(f"data will be saved in {out_path}")

    positive_filtered = [task for task in scores if scores[task]["pos_score"] != 'ERR' and scores[task]["pos_score"] > threshold]

    with open(out_path, 'wt') as f:
        str_to_dump = json.dumps(positive_filtered, indent=2)
        f.write(str_to_dump)
    print(f"{len(positive_filtered)} marked positive from {path_to_scores_file}")
    print(f"positive samples saved in {out_path}")

    pos_scores = [np.array([scores[res]["pos_score"] for res in scores if scores[res]["pos_score"] != 'ERR'],
                           dtype=np.float32)
                  ]
    plt.violinplot(pos_scores, showmeans=True)
    plt.xticks([1], labels=[path_to_scores_file])
    path_to_vis = path_to_scores_file.replace(".json", "_pos_score_dist.png")
    plt.savefig(path_to_vis)
    print(f"distribution plot in {path_to_vis}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--percentile", type=float, default=-1.0)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--out_dir")
    parser.add_argument("--scores")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.threshold == -1 and args.percentile == -1:
        raise RuntimeError("at least one of threshold or percentile should be set")
    run(args.scores, args.out_dir, args.percentile, args.threshold)

