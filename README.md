# WildIFEval

Code for **WildIFEval: Instruction Following in the Wild** ([paper][paper], [dataset][dataset]).

WildIFEval is a benchmark of 7,523 natural instruction-following tasks mined from
[LMSYS-Chat-1M][lmsys]. Each task is paired with an LLM-produced decomposition
into up to 8 atomic constraints, enabling per-constraint evaluation via an
LLM-as-a-judge. This repo holds the code to reproduce the curation pipeline, run
inference for new models, score predictions, and regenerate the paper's figures
and tables.

Dataset: <https://huggingface.co/datasets/gililior/wild-if-eval>
Paper: <https://arxiv.org/abs/2503.06573>

## Install

```sh
git clone https://github.com/gililior/wild-if-eval-code.git
cd wild-if-eval-code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit and set OPENAI_API_KEY (and HF_TOKEN if uploading)
```

The `vllm` dependency requires a CUDA GPU. If you only intend to run API
inference and analysis, you can install everything else and skip vLLM:

```sh
pip install -r requirements.txt --no-deps  # then re-install minus vllm
```

## Quickstart

### Load the dataset

```python
from datasets import load_dataset

# Load the decomposition dataset
decomposition_ds = load_dataset("gililior/wild-if-eval", split="test")

# Load and filter the original dataset
orig_ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
conversation_ids = set(decomposition_ds["conversation_id"])
orig_ds_filtered = orig_ds.filter(lambda x: x['conversation_id'] in conversation_ids)

# Keep only the first request in each conversation
def leave_only_first_request(example):
  example["conversation"] = example["conversation"][0]["content"]
  return example

orig_ds_cleaned = orig_ds_filtered.map(leave_only_first_request)
orig_ds_cleaned = orig_ds_cleaned.rename_column("conversation", "task")

# Convert decomposition dataset into a dictionary for fast lookup
decomposition_dict = {row["conversation_id"]: row for row in decomposition_ds}

# Merge decomposition with original dataset
def merge_examples(example):
    match = decomposition_dict.get(example["conversation_id"], {})
    return {**example, **match}

merged_dataset = orig_ds_cleaned.map(merge_examples)
```

The same snippet is exposed as `scripts.evaluate_llms.utils.load_data`.

### Run inference on a model

Local vLLM (GPU):

```sh
python -m scripts.evaluate_llms.local_gpu_inference \
  --dataset gililior/wild-if-eval \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --out_path model_predictions/Llama-3.1-8B-0shot-wild-if-eval.json
```

OpenAI-compatible API:

```sh
export OPENAI_API_KEY=sk-...
python -m scripts.evaluate_llms.api_inference \
  --data gililior/wild-if-eval \
  --model gpt-4o-2024-08-06 \
  --out_dir model_predictions/
```

For other OpenAI-compatible endpoints (Together, Anyscale, vLLM serve, IBM
RITS, …) pass `--API_endpoint` and `--API_key_name`. The script auto-strips
RITS-specific headers when the URL doesn't contain "rits".

### Score predictions with LLM-as-a-judge

```sh
python -m scripts.evaluate_llms.llms_aaj_constraint_multiproc \
  --data gililior/wild-if-eval \
  --to_eval model_predictions/Llama-3.1-8B-0shot-wild-if-eval.json \
  --eval_model deepseek-ai/DeepSeek-V3 \
  --out_dir llm_aaj_scores/
```

### Reproduce the paper's main results

Fetch the published predictions/scores (instead of re-running 14 models):

```sh
python -m scripts.download_predictions  # pulls gililior/wild-if-eval-predictions
```

Then regenerate figures and tables:

```sh
python -m scripts.data_analysis.plots_for_paper          # main paper figures (analysis_output/figures/)
python -m scripts.scale_experiment.aggregate_scores      # builds per_task_scores.parquet
python -m scripts.scale_experiment.run_subsample         # 50-replicate stability study
python -m scripts.scale_experiment.make_figures          # writes figures/scale_experiment/*.pdf
python -m scripts.scale_experiment.category_coverage     # per-category coverage at each N
```

## Repository structure

```
wild-if-eval-code/
├── scripts/
│   ├── arena_filtering/        # 5-stage curation pipeline (heuristic → classify → filter → decompose → upload)
│   ├── evaluate_llms/          # vLLM + OpenAI-compatible inference; LLM-as-a-judge
│   ├── data_analysis/          # paper figures, judge-agreement, length probes
│   ├── scale_experiment/       # subsampling stability + category coverage
│   └── download_predictions.py # fetch paper predictions/scores from HF
├── data/                       # small CSV/JSON inputs (constraint categories, domains, atomic-constraint list)
├── figures/                    # paper figures (committed)
│   ├── paper/                  # main-text figures (PNG)
│   └── scale_experiment/       # scale-experiment figures (PDF + PNG) and RESULTS.md
├── analysis_output/            # gitignored; runtime output dir for plots_for_paper.py
├── requirements.txt
├── .env.example
└── LICENSE
```

## Reproducing curation from scratch

Each step is independent. Output of step *N* feeds step *N+1*.

```sh
# 1. Heuristic filter (non-English, code, toxicity)
python -m scripts.arena_filtering.heuristic_filtering --out_path filtered_ids.json

# 2. Classify each task for "is this a constrained-generation request?"
python -m scripts.arena_filtering.classify_constrained_generation_tasks \
  --path_to_filtered_ids filtered_ids.json \
  --out_dir classification_scores/ \
  --classification_model meta-llama/Llama-3.1-405B-Instruct

# 3. Keep top-percentile positives
python -m scripts.arena_filtering.filter_tasks_given_pos_score \
  --scores classification_scores/constrained-gen-pos-score-llama3.1-405b.json \
  --percentile 50 \
  --out_dir filtered_tasks/

# 4. Decompose each kept task into atomic constraints
python -m scripts.arena_filtering.decompose_tasks \
  --positive_tasks filtered_tasks/filtered_0.5percentile_*.json \
  --out decomposed.json \
  --decompose_model gpt-4o-2024-08-06

# 5. Push to the HuggingFace Hub
HF_TOKEN=hf_... python -m scripts.arena_filtering.upload_data_to_hf \
  --decomposition decomposed.json \
  --name_in_hub <your-org>/wild-if-eval-rerun
```

## Citation

```bibtex
@article{lior2025wildifeval,
  title={Wildifeval: Instruction following in the wild},
  author={Lior, Gili and Yehudai, Asaf and Gera, Ariel and Ein-Dor, Liat},
  journal={arXiv preprint arXiv:2503.06573},
  year={2025}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). Same license as the underlying
[dataset on HuggingFace][dataset].

## Contact

Issues and pull requests welcome. For questions, open an issue or email
`gili.lior96@gmail.com`.

[paper]: https://arxiv.org/abs/2503.06573
[dataset]: https://huggingface.co/datasets/gililior/wild-if-eval
[lmsys]: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
