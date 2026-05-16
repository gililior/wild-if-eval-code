"""Download paper model predictions and LLM-as-a-judge scores from the
companion HuggingFace dataset and lay them out in the locations the analysis
scripts expect.

The companion dataset is `gililior/wild-if-eval-predictions`. It mirrors the
two directories the analysis scripts read from:

    model_predictions/<MODEL>-0shot-wild-if-eval.json
    llm_aaj_scores/llm-aaj-<JUDGE>.<MODEL>-0shot-wild-if-eval.json
    llm_aaj_scores/other_judges/llm-aaj-<JUDGE>.<MODEL>-0shot-wild-if-eval.json

Usage:
    python -m scripts.download_predictions
    python -m scripts.download_predictions --repo gililior/wild-if-eval-predictions --dest .
"""
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = ArgumentParser()
    parser.add_argument("--repo", default="gililior/wild-if-eval-predictions",
                        help="HuggingFace dataset repo holding the predictions/scores")
    parser.add_argument("--dest", default=".",
                        help="repo-root destination (default: current dir)")
    args = parser.parse_args()

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} into {dest} ...")
    # Restrict to the two data folders so the dataset's own README.md /
    # .gitattributes don't overwrite files in the code repo when --dest is ".".
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["model_predictions/*", "llm_aaj_scores/*"],
    )
    print("Done. Verify model_predictions/ and llm_aaj_scores/ now contain the JSON files.")


if __name__ == "__main__":
    main()
