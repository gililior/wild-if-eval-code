# Preprocessing Scripts for WildIFEval

This directory contains scripts for preprocessing the  
<a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m" target="_blank">lmsys-chat-1m</a> dataset  
to create the **WildIFEval** dataset.

## 🛠️ Preprocessing Steps

1. **Heuristic Filtering** – Removes non-English, code-based, and toxic language responses.
2. **Classification** – Uses an LLM to identify constrained generation tasks.
3. **Task Filtering** – Selects tasks based on classification scores.
4. **Task Decomposition** – Breaks down tasks into constraints using an LLM.
5. **Upload to Hugging Face** – Saves the final dataset to the Hugging Face Hub.

---

## ⚡ Running via API

Both **classification** and **decomposition** require an **LLM API**.  
To use them, specify:
- **API Key Environment Variable**: `--API_key_name`
- **API Endpoint**: `--API_endpoint`

The script retrieves the API key from the specified environment variable using:
```python
os.environ.get(API_key_name)
```

---

## 🚀 Usage

### **Step 1: Heuristic Filtering**
Removes non-English, code, and toxic responses.
```sh
python arena_filtering/heuristic_filtering.py --out_path /path/to/save/filtered/ids/json
```

### **Step 2: Classify Constrained Generation Tasks**
Uses an LLM to classify tasks.
```sh
python arena_filtering/classify_constrained_generation_tasks.py \
  --path_to_filtered_ids /path/to/save/filtered/ids/json \
  --out_dir /out/dir/to/save/classification/scores \
  --classification_model MODEL_NAME \
  --API_key_name ENV_VAR_NAME \
  --API_endpoint ENDPOINT
```

### **Step 3: Filter Tasks Based on Classification Scores**
Filters tasks using a percentile or threshold.
```sh
python arena_filtering/filter_tasks_given_pos_score.py \
  [--percentile PERCENTILE] \
  [--threshold THRESHOLD] \
  --out_dir /path/to/save/filtered/tasks/json \
  --scores /path/to/classification/scores/json
```

### **Step 4: Decompose Tasks into Constraints**
Uses an LLM to decompose tasks.
```sh
python arena_filtering/decompose_tasks.py \
  --positive_tasks /path/to/save/filtered/tasks/json \
  --out /path/to/save/decomposed/tasks/json \
  --decompose_model MODEL_NAME \
  --API_key_name ENV_VAR_NAME \
  --API_endpoint ENDPOINT
```

### **Step 5: Upload to Hugging Face Hub**
Uploads the final dataset to the Hugging Face Hub.
```sh
python arena_filtering/upload_data_to_hf.py \
  --decomposition /path/to/save/decomposed/tasks/json \
  --name_in_hub NAME_IN_HUB
```

---

## 📢 Notes
- **Ensure your API key and endpoint are correctly set** before running classification and decomposition scripts.
- **Modify model names and API details** as needed based on your setup.
- **Output directories should exist** before running the scripts.

---
