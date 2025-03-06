

In this directory are scripts that preprocess the [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) dataset.

It icludes the following steps:
1. Heuristic filtering - non English, code, toxic language.
2. Classification of constrained generation tasks using an LLM.
3. Filtering tasks based on the classification scores.
4. Decomposition of tasks into constraints using an LLM.
5. Upload the decomposed tasks to the Hugging Face Hub.

### Run via API
Both classification and decomposition assume the usage of LLM via API.
You are expected to specify the environment variable storing the API key (`--API_key_name`) 
and the API endpoint (`--API_endpoint`). The script retrieves the API key from the specified 
environment variable using `os.environ.get(API_key_name)`.

### Usage

```shell
python arena_filtering/heuristic_filtering.py --out_path /path/to/save/filtered/ids/json
```


```shell
python arena_filtering/classify_constrained_generation_tasks.py \
  --path_to_filtered_ids /path/to/save/filtered/ids/json \
  --out_dir /out/dir/to/save/classification/scores \
  --classification_model NAME \
  --API_key_name ENV_VAR_NAME \
  --API_endpoint ENDPOINT 
```

```shell
python arena_filtering/filter_tasks_given_pos_score.py \
  [--percetile PERCENTILE] \
  [--threshold THRESHOLD] \
  --out_dir /path/to/save/filtered/tasks/json \
  --scores /path/to/classification/scores/json
```

```shell
python arena_filtering/decompose_tasks.py \
  --positive_tasks /path/to/save/filtered/tasks/json \
  --out /path/to/save/decomposed/tasks/json \
  --decomompose_model MODEL_NAME \
  --API_key_name ENV_VAR_NAME \
  --API_endpoint ENDPOINT
```

```shell
python arena_filtering/upload_data_to_hf.py \
  --decompostion /path/to/save/decomposed/tasks/json \
  --name_in_hub NAME_IN_HUB
```
