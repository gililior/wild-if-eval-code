

```shell
python arena_filtering/heuristic_filtering.py --out_path /path/to/save/filtered/ids/json
```

```shell
python arena_filtering/classify_constrained_generation_tasks.py --path_to_filtered_ids /path/to/save/filtered/ids/json \
    --out_dir /out/dir/to/save/classification/scores --classification_model NAME --API_key_name ENV_VAR_NAME \
    --API_endpoint ENDPOINT 
```

```shell
python arena_filtering/filter_tasks_given_pos_score.py [--percetile PERCENTILE] [--threshold THRESHOLD] \
  --out_dir /path/to/save/filtered/tasks/json --scores /path/to/classification/scores/json
```

```shell
python arena_filtering/decompose_tasks.py --positive_tasks /path/to/save/filtered/tasks/json \
    --out /path/to/save/decomposed/tasks/json --decomompose_model MODEL_NAME \
    --API_key_name ENV_VAR_NAME --API_endpoint ENDPOINT
```

```shell
python arena_filtering/upload_data_to_hf.py --decompostion /path/to/save/decomposed/tasks/json \
    --name_in_hub NAME_IN_HUB
```


