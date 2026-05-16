# Evaluating LLM Performance on Constrained Generation Tasks

This directory contains scripts to evaluate the performance of LLMs on **constrained generation tasks**.  
Inference can be performed either **locally on a GPU** or **via an API**.

---

## 🚀 Running Inference

### **Option 1: Run Locally on GPU**
To perform inference using a local **GPU**, use the following command:

```sh
python evaluate_llms/local_gpu_inference.py --dataset gililior/wild-if-eval \
  --model MODEL_NAME_IN_HF --out_path /path/to/save/predictions/json
```

### **Option 2: Run via API**
To perform inference using an **API**, use the following command:

```sh
python evaluate_llms/api_inference.py --dataset gililior/wild-if-eval \
  --model MODEL_NAME_IN_API_ENDPOINT --out_path /path/to/save/predictions/json \
  --API_key_name ENV_VAR_NAME --API_endpoint ENDPOINT
```

### **API Requirements**
For API-based inference, you must specify:
- **API Key Environment Variable**: `--API_key_name`
- **API Endpoint**: `--API_endpoint`

The script retrieves the API key using:
```python
os.environ.get(API_key_name)
```

---

## 📊 LLM as a Judge Evaluation

Once the LLM-generated predictions are available, you can evaluate their performance using an LLM **as a judge**.

Run the following command:

```sh
python evaluate_llms/llms_aaj_constraint_multiproc.py --data gililior/wild-if-eval \
  --to_eval /path/to/predictions/json --out_path /path/to/save/evaluation/json --out_dir OUT_DIR \
  --eval_model MODEL_NAME_IN_API_ENDPOINT --API_key_name ENV_VAR_NAME --API_endpoint ENDPOINT \
  [--sample SAMPLE] [--tasks_batch_size TASKS_BATCH_SIZE] 
```

### **Evaluation Parameters**
- `--to_eval`: Path to the LLM-generated predictions.
- `--out_path`: Path to save the evaluation results.
- `--eval_model`: The model used for evaluation (must be accessible via API).
- `--API_key_name`: Name of the environment variable storing the API key.
- `--API_endpoint`: API endpoint for evaluation.
- `--sample` *(Optional)*: Number of samples to evaluate.
- `--tasks_batch_size` *(Optional)*: Batch size for evaluation.

---

## 📢 Notes
- **Ensure your API key and endpoint are correctly set** before running API-based inference or evaluation.
- **Modify model names and API details** based on your setup.
- **Ensure that output directories exist** before running the scripts.

---
