# WildIFEval: Supporting Code

This repository contains the supporting code for the <a href="https://huggingface.co/datasets/gililior/wild-if-eval" target="_blank">WildIFEval dataset</a>, which was introduced in the paper <a href="PAPER_LINK_HERE" target="_blank">"WildIFEval: Instruction Following in the Wild"</a>.

## Repository Contents

### 🛠️ **Scripts**
This repo includes the following scripts:

1. **Preprocessing** – Filters the `lmsys-chat-1m` (Chatbot Arena) dataset to create WildIFEval  
   📂 [arena_filtering/](arena_filtering)
   
2. **Inference** – Evaluates the performance of LLMs on WildIFEval  
   📂 [evaluate_llms/](evaluate_llms)

3. **Evaluation** – Uses LLMs as judges to assess performance on WildIFEval  
   📄 [evaluate_llms/llms_aaj_constraint_multiproc.py](evaluate_llms/llms_aaj_constraint_multiproc.py)

4. **Analysis** – Examines the characteristics of WildIFEval and LLM performance  
   📂 [analysis/](analysis)

**🔹 To replicate the results from the paper, follow the instructions in the respective directories.**

---

### 📊 **Data**
This repo also includes the following data:

1. **Model Predictions** – Few open-source LLMs' responses on WildIFEval  
   📂 [model_predictions/](model_predictions)

2. **Evaluation Results** – Scores obtained using LLM as a judge  
   📂 [llm_aaj_scores/](llm_aaj_scores)

3. **Analysis Output** – Results from analysis scripts, including all figures from the paper  
   📂 [analysis_output/](analysis_output)

---

### 📢 Notes
- **Links to GitHub directories and files** will open in the same tab (GitHub limitation).
- **External links** (like Hugging Face or paper links) will open in a new tab **only when viewed in a browser that supports HTML links inside Markdown.**
- If viewing on **GitHub**, you can **Ctrl + Click (Windows/Linux)** or **Cmd + Click (Mac)** to open in a new tab.

---
