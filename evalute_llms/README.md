
This directory consists of scripts to evaluate the performance of LLMs on the constrained generation tasks. 
The inference can either run on local gpu, or via API.

To run on local gpu:

```shell
python evaluate_llms/local_gpu_inference.py --dataset gililior/wild-if-eval \
  --model MODEL_NAME_IN_HF --out_path /path/to/save/predictions/json
```

To run via API:

```shell
python evaluate_llms/api_inference.py --dataset gililior/wild-if-eval \
  --model MODEL_NAME_IN_API_ENDPOINT --out_path /path/to/save/predictions/json \
  --API_key_name ENV_VAR_NAME --API_endpoint ENDPOINT
```

To run the via API, specify the environment variable storing the API key (`--API_key_name`) 
and the API endpoint (`--API_endpoint`). The script retrieves the API key from the specified 
environment variable using `os.environ.get(API_key_name)`.