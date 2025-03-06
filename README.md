This repo is the supporting code for the [WildIFEval](https://huggingface.co/datasets/gililior/wild-if-eval) dataset, 
which was presented in the paper ["WildIFEval: Instruction Following in the Wild"]().

This repo includes the following scripts:
1. Preprocessing scripts, filtering the lmsys-chat-1m (chatbot-arena) dataset to create the WildIFEval dataset - [arena_filtering/](arena_filtering)
2. Inference scripts to evaluate the performance of LLMs on WildIFEval - [evaluate_llms/](evaluate_llms)
3. Evaluation script to evaluate the performance on WildIFEval, using LLM as a Judge - [evaluate_llms/llms_aaj_constraint_multiproc.py](evaluate_llms/llms_aaj_constraint_multiproc.py)
4. Analysis scripts to analyze the characteristics of WildIFEval, and also the performance of LLMs on WildIFEval - [analysis/](analysis)

**To replicate the results from the paper, you can follow the instructions in the respective directories.**


This repo also includes the following data:
1. Few open-source LLMs predictions on WildIFEval - [model_predictions/](model_predictions)
2. The evaluation results using LLM as a Judge on the existing predictions - [llm_aaj_scores/](llm_aaj_scores)
3. Output of the analysis scripts, including all figures from the paper - [analysis_output/](analysis_output)