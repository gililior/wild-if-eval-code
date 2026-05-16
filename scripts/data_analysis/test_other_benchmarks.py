
from datasets import load_dataset


if_eval = load_dataset('google/IFEval', split='all')
info_bench = load_dataset('kqsong/InFoBench', split='all')

constraints_infobench = info_bench["decomposed_questions"]
constraints_if_eval = if_eval["instruction_id_list"]

info_bench_flatten = [const for sublist in constraints_infobench for const in sublist]
print("number of unique constraints in InFoBench", len(set(info_bench_flatten)))

if_eval_flatten = [const for sublist in constraints_if_eval for const in sublist]
print("number of unique constraints in IFEval", len(set(if_eval_flatten)))