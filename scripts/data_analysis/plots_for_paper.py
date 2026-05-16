import os.path
from pathlib import Path
import numpy as np
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.manifold import TSNE
from scripts.data_analysis.utils import generate_color_map
from scripts.evaluate_llms.utils import load_data
import gdown
import random
from datasets import Dataset
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]

DS_NAME = "gililior/wild-if-eval-decomposed-gpt4"
MIN_NUM_CONSTRAINTS = 2
MAX_NUM_CONSTRAINTS = 8
CONSTS_RANGE = (MIN_NUM_CONSTRAINTS, MAX_NUM_CONSTRAINTS+1)

GENERATIONS_DIR = str(REPO_ROOT / "model_predictions")
DATA_DIR = str(REPO_ROOT / "data")
ANALYSIS_DIR = str(REPO_ROOT / "analysis_output")
CONSTRAINT_CATEGORY_CLASSIFICATION = os.path.join(DATA_DIR, "constraint_categories.csv")
DOMAINS_PATH = os.path.join(DATA_DIR, "domains.csv")
ATOMICS_EMBEDDINGS = os.path.join(ANALYSIS_DIR, "atomics.npy")
ATOMICS_SORTED = os.path.join(DATA_DIR, "sorted_atomics.json")


SCORES_DIR = str(REPO_ROOT / "llm_aaj_scores")
SCORES_FNAME = "llm-aaj-deepseek-v3.{model_name}-0shot-wild-if-eval.json"
PATH_TO_SCORES = os.path.join(SCORES_DIR, SCORES_FNAME)

OUT_DIR_FOR_PLOTS = str(REPO_ROOT / "analysis_output" / "figures")
os.makedirs(OUT_DIR_FOR_PLOTS, exist_ok=True)


def sort_model_name(word):
    first_letter = word[0]  # First letter
    try:
        middle_letter = float(word.split('-')[-1][:-1])  # Middle letter
    except ValueError:
        middle_letter = 0
    return first_letter, middle_letter  # Sorting by first, then middle letter

def filter_constraints(ds):
    # filter out tasks with a single constraint
    ds_filtered = ds.filter(lambda x: len(x["decomposition"]) >= MIN_NUM_CONSTRAINTS)
    # with open("tasks_to_remove.json", 'r') as f:
        # tasks_to_remove = json.load(f)
    # ds_filtered = ds_filtered.filter(lambda x: x['task'] not in tasks_to_remove)
    return ds_filtered


def calculate(paths):

    color_map_by_model = generate_color_map()
    all_jsons = {}
    ds = load_data(DS_NAME)
    ds = filter_constraints(ds)

    print("number of tasks", len(ds))

    all_constraints = ds["decomposition"]
    all_lengths = [len(c) for c in all_constraints]
    print("mean number of constraints", np.mean(all_lengths))

    all_constraints = [item for sublist in all_constraints for item in sublist]
    all_unique_constraints = set(all_constraints)
    print("number of unique constraints", len(all_unique_constraints))
    print("num of tasks", len(ds["task"]))

    plot_frequency_of_constraints(all_constraints)
    plot_histogram_of_num_constraints_in_task(all_lengths)

    domain_df = pd.read_csv(DOMAINS_PATH)
    domain_df = domain_df[domain_df["task"].isin(ds["task"])]
    domain_count = Counter(domain_df["domain"])
    domain_count.pop("Artificial Intelligence")
    ordered_categories = sorted(domain_count.keys(), key=lambda x: domain_count[x], reverse=True)
    ordered_counts = [domain_count[cat] for cat in ordered_categories]
    ordered_categories = [cat if not pd.isna(cat) else "Other" for cat in ordered_categories]

    plot_domains_distribution(ordered_categories, ordered_counts)

    for k in paths:
        with open(paths[k], 'r') as f:
            preds = json.load(f)
        all_jsons[k] = {task: preds[task] for task in ds["task"] if task in preds}

    all_scores, all_constraints_scores, constraint_to_category = get_all_scores(all_jsons, ds, domain_df)

    # get for each model, the constraints that get the lowest scores, and extract the task from it
    all_indices = {}
    for model in all_jsons.keys():
        constraint_scores = all_constraints_scores[model]
        # get all indices of rows that got 0
        indices = constraint_scores[constraint_scores["binary_score"] == 0].index
        all_indices[model] = indices
    # get the intersection of all indices
    common_indices = set.intersection(*[set(all_indices[k]) for k in all_indices])
    common_indices = np.array(list(common_indices))

    for model in ["Deepseek-v3", "Llama3.3-70b", "Mistral-large", "Llama3.1-405b"]:
        all_other_model_wrong_indices = set.intersection(*[set(all_indices[k]) for k in all_indices if k != model]).difference(common_indices)
        all_other_model_wrong_indices = np.array(list(all_other_model_wrong_indices))
        print(model, ":", len(all_other_model_wrong_indices), "constraints in which all other models failed")
        all_other_as_df = all_constraints_scores[model].iloc[all_other_model_wrong_indices]
        grooup_by_task = all_other_as_df.groupby("orig_task")
        all_other_as_json = {}
        for group in grooup_by_task:
            task = group[0]
            group_df = group[1]
            all_other_as_json[task] = {"task": task, "total_num_constraints": int(group_df["total_constraints"].max())}
            all_other_as_json[task]["num_failed"] = len(group_df)
            consts = group_df["constraint"].to_list()
            cat = group_df["category"].to_list()
            failed_constraints = {consts[i]: cat[i] for i in range(len(consts))}
            all_other_as_json[task]["failed_constraints"] = failed_constraints
        with open(f"all_others_faild_{model}.json", 'w') as f:
            json.dump(all_other_as_json, f, indent=2)

    all_big_succeeds_but_rest_fail = {}
    succeed_indices = {}
    for model in ["Deepseek-v3", "Llama3.3-70b", "Mistral-large", "Llama3.1-405b"]:
        succeed_indices[model] = all_constraints_scores[model][all_constraints_scores[model]["binary_score"] == 1].index
    all_big_succeeds = set.intersection(*[set(succeed_indices[k]) for k in succeed_indices])
    all_big_succeeds_but_small_fail = all_big_succeeds.copy()
    for model in all_jsons.keys():
        if model in ["Deepseek-v3", "Llama3.3-70b", "Mistral-large", "Llama3.1-405b"]:
            continue
        all_model_succeeds = all_constraints_scores[model][all_constraints_scores[model]["binary_score"] == 1].index
        all_big_succeeds_but_small_fail = all_big_succeeds_but_small_fail.difference(all_model_succeeds)
    print(all_big_succeeds_but_small_fail)

    df_to_json = all_constraints_scores["Deepseek-v3"].iloc[np.array(list(all_big_succeeds_but_small_fail))]
    group_by_task = df_to_json.groupby("orig_task")
    to_json = {}
    for group in group_by_task:
        task = group[0]
        group_df = group[1]
        to_json[task] = {"task": task, "total_num_constraints": int(group_df["total_constraints"].max())}
        to_json[task]["num_failed"] = len(group_df)
        consts = group_df["constraint"].to_list()
        cat = group_df["category"].to_list()
        failed_constraints = {consts[i]: cat[i] for i in range(len(consts))}
        to_json[task]["failed_constraints"] = failed_constraints
    to_json = list(to_json.values())
    with open(f"big_vs_small.json", 'w') as f:
        json.dump(to_json, f, indent=2)



    for model in all_jsons.keys():
        constraint_scores_only_hard = all_constraints_scores[model].iloc[np.array(list(common_indices))]
        # make sure that all binary scores are 0
        assert all(constraint_scores_only_hard["binary_score"] == 0)
        # get the tasks for these indices

    print("there are", len(common_indices), "hard constraints")
    num_constraints_in_hard_tasks = constraint_scores_only_hard.groupby("orig_task")["total_constraints"].max()
    plot_histogram_of_num_constraints_in_task(num_constraints_in_hard_tasks.to_list(), "_only_hard")

    hard_json = {}
    group_by_task = constraint_scores_only_hard.groupby("orig_task")
    for task, group in group_by_task:
        orig_num_constraints = group["total_constraints"].max()
        hard_json[task] = {"task": task, "total_num_constraints": int(orig_num_constraints)}
        hard_json[task]["num_failed"] = len(group)
        consts = group["constraint"].to_list()
        cat = group["category"].to_list()
        failed_constraints = {consts[i]: cat[i] for i in range(len(consts))}
        hard_json[task]["failed_constraints"] = failed_constraints
    with open("hard_tasks.json", 'w') as f:
        json.dump(hard_json, f, indent=2)
    print("there are ", len(hard_json), "hard tasks")

    hard_tasks_as_list = list(hard_json.values())
    with open("hard_tasks_as_list.json", 'w') as f:
        json.dump(hard_tasks_as_list[:20], f, indent=2)


    # randomly pick tasks from the dataset, the same number as the number of tasks in the hard tasks
    random_rows = ds.shuffle().select(range(len(hard_json)))
    random_json = {}
    for row in random_rows:
        random_json[row["task"]] = {"task": row["task"],
                                    "total_num_constraints": len(row["decomposition"]),
                                    "all_constraints": row["decomposition"],
                                    "all_categories": [constraint_to_category[const] for const in row["decomposition"]]}
    with open("random_tasks.json", 'w') as f:
        json.dump(random_json, f, indent=2)
    random_as_list = []
    for task in random_json:
        if task not in domain_df["task"].to_list():
            continue
        val = random_json[task]
        consts = random_json[task]["all_constraints"]
        categories = random_json[task]["all_categories"]
        constraints = {consts[i]:categories[i] for i in range(len(consts))}
        random_as_list.append({
            "task": val["task"],
            "domain": domain_df[domain_df["task"] == val["task"]]["domain"].values[0],
            "total_num_constraints": val["total_num_constraints"],
            "constraints": constraints,
        })
    with open("random_as_list.json", 'w') as f:
        json.dump(random_as_list[:30], f, indent=2)


    # get tasks that all models achieve perfect scores
    # all_fulfilled_tasks = {}
    # for model in all_jsons.keys():
    #     all_fulfilled_tasks[model] = all_scores[model][all_scores[model]["all_fulfilled"] == 1].index
    # # get the intersection of all indices
    # common_fulfilled_indices = set.intersection(*[set(all_fulfilled_tasks[k]) for k in all_fulfilled_tasks])
    # for model in all_jsons.keys():
    #     constraint_scores_only_fulfilled = all_scores[model].iloc[np.array(list(common_fulfilled_indices))]
    #     # make sure that all binary scores are 1
    #     assert all(constraint_scores_only_fulfilled["all_fulfilled"] == 1)
    #     # get the tasks for these indices
    # print("number of tasks that all models achieved perfect scores", len(common_fulfilled_indices))
    # tasks_to_remove = np.array(ds["task"])[np.array(list(common_fulfilled_indices))].tolist()
    # with open('tasks_to_remove.json', 'w') as f:
    #     json.dump(tasks_to_remove, f, indent=2)

    categories_sorted = sorted(set(constraint_to_category.values()))
    colors_for_categories = {c: plt.cm.tab20.colors[i * 2] for i, c in enumerate(categories_sorted)}



    # plot_constraint_embeddings(all_unique_constraints, constraint_to_category, categories_sorted, colors_for_categories)

    models_sorted = sorted(all_scores.keys(), key=sort_model_name)
    models_sorted.reverse()

    model_colors = [color_map_by_model[model] for model in models_sorted]

    #  count how many constraints are in each category
    count = Counter(constraint_to_category.values())
    category_constraint_labels = list(count.keys())
    category_constraint_labels = sorted(category_constraint_labels, key=lambda x: count[x], reverse=True)

    category_frequency_for_hard_tasks = constraint_scores_only_hard["category"].value_counts().to_dict()
    plt.figure("frequency of categories only hard", figsize=(6, 6))
    plt.bar(category_constraint_labels, [category_frequency_for_hard_tasks[cat] for cat in category_constraint_labels],
            color=[colors_for_categories[cat] for cat in category_constraint_labels], edgecolor='black',
            alpha=0.9)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "frequency_of_categories_hard.png"))

    plot_category_frequency(all_constraints_scores, category_constraint_labels, colors_for_categories, count)

    category_constraint_labels.remove("Other")  # remove "Other" category

    plot_co_occurrences_scores(all_constraints_scores['Deepseek-v3'], category_constraint_labels)

    plot_co_occurrences(category_constraint_labels, constraint_to_category, domain_df, ds)

    plot_performance_as_function_of_num_constraints(all_constraints_scores, all_scores, model_colors, models_sorted)

    means = [all_scores[model]["all_fulfilled"].mean() for model in models_sorted]
    plot_mean_scores(means, model_colors, models_sorted, "strict")

    means_soft = [all_scores[model]["mean_score"].mean() for model in models_sorted]
    plot_mean_scores(means_soft,  model_colors, models_sorted, "soft")

    from statsmodels.stats.contingency_tables import mcnemar
    from itertools import combinations

    # Extract model names

    # Prepare matrix to store p-values
    pvals = pd.DataFrame(index=models_sorted, columns=models_sorted)

    # For each model pair, compute McNemar's test
    for model_a, model_b in combinations(models_sorted, 2):
        preds_a = np.array(all_scores[model_a]["all_fulfilled"])
        preds_b = np.array(all_scores[model_b]["all_fulfilled"])

        # Contingency table
        both_correct = np.sum((preds_a == 1) & (preds_b == 1))
        a_only = np.sum((preds_a == 1) & (preds_b == 0))
        b_only = np.sum((preds_a == 0) & (preds_b == 1))
        both_wrong = np.sum((preds_a == 0) & (preds_b == 0))

        table = [[both_correct, a_only],
                 [b_only, both_wrong]]

        result = mcnemar(table, exact=False, correction=True)
        pval = result.pvalue

        # Fill in both (i,j) and (j,i) for symmetry
        pvals.loc[model_a, model_b] = pval
        pvals.loc[model_b, model_a] = pval

    # Fill diagonal with NaN or 1.0
    np.fill_diagonal(pvals.values, np.nan)

    # Optional: convert to float
    pvals = pvals.astype(float)

    def mask_upper_triangle(df):
        mask = np.triu(np.ones(df.shape), k=1).astype(bool)
        return mask

    def format_pval(val):
        try:
            val = float(val)
            if val < 0.01:
                return '*'
            else:
                return f"{val:.3f}"
        except:
            return ''

    # Apply mask to hide upper triangle
    mask = mask_upper_triangle(pvals)

    # Create annotation matrix
    annot = pvals.copy().applymap(format_pval)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pvals,
        annot=annot,
        mask=mask,
        cmap="viridis_r",
        cbar_kws={"label": "p-value"},
        fmt="",
        linewidths=0.5,
        linecolor='white',
        square=True
    )
    plt.title("Pairwise McNemar p-values (lower triangle only)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    from scipy.stats import ttest_rel

    pvals = pd.DataFrame(index=models_sorted, columns=models_sorted)

    # Compute pairwise p-values
    for model_a, model_b in combinations(models_sorted, 2):
        x = np.array(all_scores[model_a]["mean_score"])
        y = np.array(all_scores[model_b]["mean_score"])

        stat, pval = ttest_rel(x, y)

        pvals.loc[model_a, model_b] = pval
        pvals.loc[model_b, model_a] = pval

    # Diagonal = NaN
    np.fill_diagonal(pvals.values, np.nan)
    pvals = pvals.astype(float)

    # Format annotation: * if significant
    def format_pval(val):
        if np.isnan(val):
            return ""
        return "*" if val < 0.01 else f"{val:.3f}"

    annot = pvals.applymap(format_pval)

    # Lower triangle mask
    mask = np.triu(np.ones(pvals.shape), k=1)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pvals,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="viridis_r",
        cbar_kws={"label": "p-value"},
        linewidths=0.5,
        linecolor='white',
        square=True
    )
    plt.title("Pairwise Paired t-test p-values (Soft Scores)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("\n\nMean scores:")
    for i, model in enumerate(models_sorted):
        print(f"{model}: {means[i]}")

    strict_means = {model: all_scores[model]["all_fulfilled"].mean().round(3).item() for model in models_sorted}
    out_nice = json.dumps(strict_means, indent=2)
    print(out_nice)

    plot_score_by_constraint_category(all_constraints_scores, category_constraint_labels,
                                      constraint_to_category, model_colors, models_sorted, colors_for_categories)



def plot_score_by_constraint_category(all_constraints_scores, category_constraint_labels, constraint_to_category,
                                      model_colors, models_sorted, colors_for_categories):
    plt.figure("bar plot of mean score by category")
    all_cat_scores = []
    for category in category_constraint_labels:
        cat_scores = []
        constraints_in_cat = [constraint for constraint in constraint_to_category if
                              constraint_to_category[constraint] == category]
        for model in models_sorted:
            mean_category_score = \
            all_constraints_scores[model].loc[all_constraints_scores[model]["constraint"].isin(constraints_in_cat)][
                "score"].mean()
            cat_scores.append(mean_category_score)
            print(f"{model} mean score for {category}: {mean_category_score}")
        all_cat_scores.append(cat_scores)
    all_cat_scores = np.array(all_cat_scores)
    # Bar width
    bar_width = 0.05
    x = np.arange(len(category_constraint_labels))  # Group positions
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models_sorted):
        ax.bar(x + i * bar_width, all_cat_scores[:, i], width=bar_width, label=model, color=model_colors[i])
    # Formatting
    ax.set_xticks(x + (bar_width * (len(models_sorted) - 1)) / 2)
    ax.set_xticklabels(category_constraint_labels, rotation=45, ha='right')
    ax.set_ylabel('Fraction of fulfilled constraints')
    # reverse the order of labels in legend
    handles, labels_for_legend = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_for_legend[::-1], bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "bar_plot_of_mean_score_by_category.png"))
    plt.figure("bar plot of mean score by category only big models")
    bar_width = 0.12
    models_to_include = ["Deepseek-v3", "Llama3.3-70b", "Mistral-large", "Llama3.1-405b", "Qwen2.5-72b"]
    reorg_categories = ["Focus / Emphasis", "Style and Tone", "Persona and Role", "Include / Avoid", "Ensure Quality",
                        "Format and Structure", "Editing", "Length"]
    x = np.arange(len(reorg_categories))  # Group positions
    reorg_categories_idx = [category_constraint_labels.index(cat) for cat in reorg_categories]
    only_big_model_scores = all_cat_scores[:, [models_sorted.index(model) for model in models_to_include]]
    only_big_model_scores = only_big_model_scores[reorg_categories_idx]
    only_big_model_colors = [model_colors[models_sorted.index(model)] for model in models_to_include]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models_to_include):
        ax.bar(x + i * bar_width, only_big_model_scores[:, i], width=bar_width, label=model,
               color=only_big_model_colors[i])
    # Formatting
    ax.set_xticks(x + (bar_width * (len(models_to_include) - 1)) / 2)
    ax.set_xticklabels(reorg_categories, rotation=45, ha='right')
    plt.ylim(0,1)
    ax.set_ylabel('Fraction of fulfilled constraints')
    ax.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=len(models_to_include))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "bar_plot_of_mean_score_by_category_only_big_models.png"))

    # plot for each model the mean score for each category
    models_categories_heatmap = np.zeros((len(category_constraint_labels), len(models_sorted)))
    for i, model in enumerate(models_sorted):
        plt.figure(f"bar plot of mean score by category {model}")
        sort_indices = all_cat_scores[:, i].argsort()[::-1]
        ordered_categories = np.array(category_constraint_labels)[sort_indices].tolist()
        for j, index in enumerate(sort_indices):
            models_categories_heatmap[index, i] = j
            score = all_cat_scores[index, i]
            plt.bar(j, score, color=colors_for_categories[category_constraint_labels[index]],
                    edgecolor='black', alpha=0.9)
        plt.xticks(range(len(ordered_categories)), ordered_categories, rotation=45, ha='right')
        plt.ylabel('Category Score')
        plt.ylim(0,1)
        plt.title(f"{model}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, f"bar_plot_of_mean_score_by_category_{model}.png"))
    mean_category_rank = models_categories_heatmap.mean(axis=1)
    argsort_by_category = np.argsort(mean_category_rank)
    models_categories_heatmap_reorg = models_categories_heatmap[argsort_by_category]
    labels_for_heatmap = np.array(category_constraint_labels)[argsort_by_category]
    plt.figure("ranking heatmap sorted", figsize=(8, 6))
    plt.imshow(models_categories_heatmap_reorg, cmap='Blues_r', aspect='auto', vmin=0, vmax=len(category_constraint_labels))
    cbar = plt.colorbar(label='Category Rank', )
    cbar.ax.invert_yaxis()
    plt.yticks(range(len(category_constraint_labels)), labels_for_heatmap, fontsize=16)
    plt.xticks(range(len(models_sorted)), models_sorted, rotation=90, fontsize=16, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "ranking_heatmap_sorted.pdf"))

    plt.figure("ranking heatmap", figsize=(8, 6))
    plt.imshow(models_categories_heatmap, cmap='Blues_r', aspect='auto', vmin=0, vmax=len(category_constraint_labels))
    plt.colorbar(label='Category Rank')
    plt.yticks(range(len(category_constraint_labels)), category_constraint_labels, fontsize=16)
    plt.xticks(range(len(models_sorted)), models_sorted, rotation=90, fontsize=16, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "ranking_heatmap.pdf"))


def plot_mean_scores(means, model_colors, models_sorted, name):
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(models_sorted[::-1], means[::-1], color=model_colors[::-1], width=0.5)
    # Add text labels with the values
    for bar in bars:
        height = bar.get_height()  # Get the height of each bar (corresponding to the value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Center text horizontally
            height + 0.01,  # Offset text slightly above the bar
            f"{height:.2f}",  # Format the value to 2 decimal places
            ha='center',  # Align horizontally
            va='bottom'  # Align vertically
        )
    plt.ylabel("Soft Score (Fraction of Fulfilled Constraints)", fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')  # Rotate labels if needed
    # plt.title(SCORES_FNAME[:SCORES_FNAME.find("{model_name}")], fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, f"bar_plot_of_mean_score_{name}.png"))


def plot_performance_as_function_of_num_constraints(all_constraints_scores, all_scores, model_colors, models_sorted):
    score_1 = {}
    per_category_means = {}
    task_scores_as_function_of_num_constraints = {}
    constraint_scores_as_function_of_num_constraints = {}
    normalized_scores_as_function_of_num_constraints = {}
    for i, model in enumerate(models_sorted):
        normalized_scores_as_function_of_num_constraints[model] = []
        mean_constraint_and_category_scores = all_constraints_scores[model].groupby(["total_constraints", "category"])["binary_score"].mean()
        for num_constraints in range(*CONSTS_RANGE):
            mean_data = mean_constraint_and_category_scores.get(num_constraints)
            normalized_scores_as_function_of_num_constraints[model].append(mean_data.mean())
        plt.figure("normalized scores as function of num constraints")
        plt.plot(range(*CONSTS_RANGE), normalized_scores_as_function_of_num_constraints[model], label=model, color=model_colors[i])

        grouped = all_constraints_scores[model].groupby("total_constraints")["binary_score"]
        means = []
        lower_bounds = []
        upper_bounds = []
        for num_constraints in range(*CONSTS_RANGE):
            if num_constraints in grouped.groups:
                scores = grouped.get_group(num_constraints)
                mean = scores.mean()
                n = len(scores)
                se = np.sqrt(mean * (1 - mean) / n)
                ci = 1.96 * se  # 95% confidence interval

                means.append(mean)
                lower_bounds.append(mean - ci)
                upper_bounds.append(mean + ci)
            else:
                means.append(np.nan)
                lower_bounds.append(np.nan)
                upper_bounds.append(np.nan)

        x_vals = list(range(*CONSTS_RANGE))

        # Plot mean line
        plt.plot(x_vals, means, label=model, color=model_colors[i])

        # Plot CI band
        plt.fill_between(x_vals, lower_bounds, upper_bounds, color=model_colors[i], alpha=0.2)

        task_scores_as_function_of_num_constraints[model] = []
        mean_task_scores = all_scores[model].groupby(["num_constraints"])["all_fulfilled"].mean()
        for num_constraints in range(*CONSTS_RANGE):
            mean_data = mean_task_scores.get(num_constraints)
            task_scores_as_function_of_num_constraints[model].append(mean_data)
        plt.figure("task scores as function of num constraints")
        plt.plot(range(*CONSTS_RANGE), task_scores_as_function_of_num_constraints[model], label=model, color=model_colors[i])

        all_categories = all_constraints_scores[model]["category"].unique()
        per_category_means[model] = {cat: [] for cat in all_categories}
        all_means_per_category = all_constraints_scores[model].groupby(["total_constraints", "category"])["binary_score"].mean()
        for num_constraints in range(*CONSTS_RANGE):
            for cat in all_categories:
                mean_data_for_cat = all_means_per_category.get((num_constraints, cat))
                per_category_means[model][cat].append(mean_data_for_cat)

        score_1[model] = task_scores_as_function_of_num_constraints[model][0]
        # plot line plot
    plt.figure("normalized scores as function of num constraints")
    plt.xlabel("Number of constraints in a task", fontsize=12)
    plt.ylabel("Normalized mean score", fontsize=12)
    figure_helper(score_1)
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "line_plot_of_normalized_score_by_num_constraints.png"))
    plt.close()

    plt.figure("constraint scores as function of num constraints")
    plt.xlabel("Number of constraints in a task", fontsize=12)
    plt.ylabel("Constraint mean score", fontsize=12)
    plt.title("Soft", fontsize=12)
    # figure_helper(score_1)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "line_plot_of_constraint_score_by_num_constraints.png"))
    plt.close()

    plt.figure("task scores as function of num constraints")
    plt.xlabel("Number of constraints in a task", fontsize=12)
    plt.ylabel("Task mean score", fontsize=12)
    plt.title("Strict", fontsize=12)
    figure_helper(score_1)
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "line_plot_of_task_score_by_num_constraints.png"))


    for model in models_sorted:
        for cat in per_category_means[model]:
            plt.figure(f"line plot by category {cat}")
            plt.plot(range(*CONSTS_RANGE), per_category_means[model][cat], label=model, color=model_colors[models_sorted.index(model)])
            plt.xlabel("Number of constraints in a task", fontsize=12)
            plt.ylabel("Fraction of fulfilled constraints", fontsize=12)
            plt.title(f"line plot of mean score by num constraints for {cat}", fontsize=12)
            plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, f"line_plot_by_category_{cat.replace(os.sep, '-').replace(' ', '-')}.png"))


def figure_helper(score_1):
    plt.xticks(range(*CONSTS_RANGE))
    handles, labels_for_figure = plt.gca().get_legend_handles_labels()
    custom_order = sorted(score_1.keys(), key=lambda x: -score_1[x])
    # Sort by custom order
    sorted_handles_labels = sorted(zip(handles, labels_for_figure), key=lambda x: custom_order.index(x[1]))
    # Unzip into sorted handles and labels
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    # Add legend
    plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()


def plot_co_occurrences_scores(all_constraints_scores, category_constraint_labels):
    mean_category_scores = all_constraints_scores.groupby(["category"])["binary_score"].mean()
    means_to_subtract = np.zeros((len(category_constraint_labels), len(category_constraint_labels)))
    for i in range(len(category_constraint_labels)):
        means_to_subtract[i] = np.full((len(category_constraint_labels),), fill_value=mean_category_scores[category_constraint_labels[i]])
    groupby_task = all_constraints_scores.groupby("orig_task")
    heatmap_category = np.zeros((len(category_constraint_labels), len(category_constraint_labels)))
    counter_cat = np.zeros((len(category_constraint_labels), len(category_constraint_labels)))
    for task, df_task in groupby_task:

        if len(df_task) < 2:
            continue
        for i, row in df_task.iterrows():
            if row["category"] not in category_constraint_labels:
                continue
            for j, col in df_task.iterrows():
                if i == j or col["category"] not in category_constraint_labels:
                    continue
                index_i_in_heatmap = category_constraint_labels.index(row["category"])
                index_j_in_heatmap = category_constraint_labels.index(col["category"])
                counter_cat[index_i_in_heatmap][index_j_in_heatmap] += 1
                counter_cat[index_j_in_heatmap][index_i_in_heatmap] += 1
                heatmap_category[index_i_in_heatmap][index_j_in_heatmap] += row["binary_score"]
                heatmap_category[index_j_in_heatmap][index_i_in_heatmap] += col["binary_score"]
    # normalize heatmap by category frequency
    norm_heatmap_category = (heatmap_category / counter_cat).round(2)
    print(norm_heatmap_category)

    print(category_constraint_labels)



def plot_co_occurrences(category_constraint_labels, constraint_to_category, domain_df, ds):
    # heatmap of co-occurances in task level
    heatmap = np.zeros((len(category_constraint_labels), len(category_constraint_labels)))
    n_tasks_for_constraints = 0
    print("calculating heatmap of co-occurrences")
    for i, constraints in tqdm(enumerate(ds["decomposition"]), total=len(ds)):
        categories = set()
        for constraint in constraints:
            constraint_label = constraint_to_category[constraint]
            if constraint_label == "Other":
                continue
            categories.add(constraint_label)
        if len(categories) >= 2:
            n_tasks_for_constraints += 1
            all_pairs = set([(i, j) for i in categories for j in categories])
            for k, j in all_pairs:
                heatmap[category_constraint_labels.index(k), category_constraint_labels.index(j)] += 1

    # normalize heatmap by category frequency

    # 1. Compute empirical probabilities
    C = heatmap.astype(float)
    N = n_tasks_for_constraints

    P_ij = C / N  # joint probabilities
    P_i = np.diag(C) / N  # marginal probabilities
    P_i_outer = np.outer(P_i, P_i)  # expected under independence

    # 2. Compute PMI matrix
    epsilon = 1e-10  # avoid log(0)
    PMI = np.log((P_ij + epsilon) / (P_i_outer + epsilon))

    # 3. Optional: zero out diagonal (PMI of (i,i) is not meaningful)
    np.fill_diagonal(PMI, 0)

    mask = np.triu(np.ones_like(PMI, dtype=bool), k=1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        PMI,
        vmin=-1, vmax=1,
        mask=mask,
        xticklabels=category_constraint_labels,
        yticklabels=category_constraint_labels,
        cmap="BrBG",  # diverging color map: blue=less, red=more than expected
        center=0,  # so white = expected
        annot=True,  # show the actual values
        fmt=".2f",
        square=True,
        cbar_kws={'label': 'PMI (log co-occurrence ratio)',},
    )
    # plt.title("Normalized Co-occurrence Heatmap (PMI)")
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "heatmap_pmi.png"))

    new_heatmap = np.zeros((len(category_constraint_labels), len(category_constraint_labels)))
    for i in range(len(category_constraint_labels)):
        for j in range(len(category_constraint_labels)):
            if i == j:
                new_heatmap[i, j] = 1
            else:
                new_heatmap[i, j] = heatmap[i, j] / (heatmap[i, i] * heatmap[j, j] / n_tasks_for_constraints)
    plt.figure("heatmap of co-occurrences in tasks")
    mask = np.triu(np.ones_like(heatmap, dtype=bool), k=1)
    masked_heatmap = np.ma.masked_where(mask, new_heatmap)
    # Plot the heatmap
    img = plt.imshow(masked_heatmap, cmap='BrBG', interpolation='nearest', vmin=0, vmax=2)
    # Set axis labels
    plt.xticks(range(len(category_constraint_labels)), category_constraint_labels, rotation=45, ha='right')
    plt.yticks(range(len(category_constraint_labels)), category_constraint_labels)
    # Add colorbar and set label
    cbar = plt.colorbar(img)
    cbar.set_label("Observed to expected ratio", fontsize=12)  # Add title to the colorbar
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "heatmap_of_co-occurrences_in_tasks.png"))


def plot_category_frequency(all_constraints_scores, category_constraint_labels, colors_for_categories, count):
    # plot frequencies
    plt.figure("frequency of categories", figsize=(6, 6))
    plt.bar(category_constraint_labels, [count[l] for l in category_constraint_labels],
            color=[colors_for_categories[cat] for cat in category_constraint_labels], edgecolor='black', alpha=0.9)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "frequency_of_categories.png"))
    for i in range(2, 9):
        distribution = Counter(
            all_constraints_scores["Deepseek-v3"].groupby('total_constraints').get_group(i)['category'].to_list())
        plt.figure(f"frequency of categories_{i}", figsize=(6, 6))
        # count how many constraints are in each category
        # plot frequencies
        plt.bar(category_constraint_labels, [distribution[l] for l in category_constraint_labels],
                color=[colors_for_categories[cat] for cat in category_constraint_labels], edgecolor='black', alpha=0.9)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title(f"{i} Constraints in a task", fontdict={'fontsize': 20})
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, f"frequency_of_categories_{i}.png"))


def download_embeddings():
    # Replace with your file ID (from Google Drive URL)
    file_id = "1b_h-yVoqjDCO4X4MGFZ3wNXWhNrnACgJ"
    url = f"https://drive.google.com/uc?id={file_id}"
    # Download the file
    gdown.download(url, ATOMICS_EMBEDDINGS, quiet=False)


def plot_constraint_embeddings(all_unique_constraints, constraint_to_category, categories_sorted, colors_for_categories):
    # load embeddings
    constraints_for_embed = []
    categories_for_embed = []
    embeddings_for_plot = []
    with open(ATOMICS_SORTED, 'r') as f:
        constraints_list = json.load(f)
    if not os.path.exists(ATOMICS_EMBEDDINGS):
        download_embeddings()
    embeddings = np.load(ATOMICS_EMBEDDINGS)
    for i, const in enumerate(constraints_list):
        if const in all_unique_constraints and const not in constraints_for_embed:
            category = constraint_to_category[const]
            if category == "Other":
                continue
            constraints_for_embed.append(const)
            categories_for_embed.append(constraint_to_category[const])
            embeddings_for_plot.append(embeddings[i])
    # plot the embeddings
    tsne = TSNE(n_components=2, random_state=0)
    # sample 1000 points
    np.random.seed(0)
    random_points = np.random.choice(range(len(embeddings_for_plot)), 1000, replace=False)
    embeddings_for_plot = [embeddings_for_plot[i] for i in random_points]
    categories_for_embed = [categories_for_embed[i] for i in random_points]
    embeddings_for_plot = np.array(embeddings_for_plot)
    embeddings_2d = tsne.fit_transform(embeddings_for_plot)
    plt.figure("tsne of embeddings", figsize=(6, 6))

    for i, category in enumerate(categories_sorted):
        indices = [j for j, cat in enumerate(categories_for_embed) if cat == category]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=category,
                    color=colors_for_categories[category], alpha=0.9)
    # set legend above plot
    plt.legend(loc='upper left', bbox_to_anchor=(-0.15, -0.05), ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "tsne_of_embeddings.png"))


def plot_domains_distribution(ordered_categories, ordered_counts):
    colors = plt.cm.Set3.colors
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(
        ordered_counts, labels=ordered_categories, autopct='%1.1f%%',
        colors=colors, startangle=0,
        wedgeprops={'edgecolor': 'white'}, textprops={'fontsize': 14},
        labeldistance=1.05,  # Moves labels closer to the pie
        pctdistance=0.85  # Moves percentage labels slightly inward
    )
    for i, text in enumerate(texts):
        text.set_bbox(dict(facecolor='none', edgecolor='none', alpha=0.75))  # Background for readability
    plt.margins(0)
    plt.tight_layout()
    output_path = os.path.join(OUT_DIR_FOR_PLOTS, "domain_piechart.png")
    plt.savefig(output_path, dpi=300)  # Save with high resolution


def plot_histogram_of_num_constraints_in_task(all_lengths, name_add=""):
    plt.figure("histogram of num constraints"+name_add)
    plt.bar(range(*CONSTS_RANGE), [all_lengths.count(i) for i in range(2, 9)], color='#ff7f0e', edgecolor='black')
    plt.xticks(range(*CONSTS_RANGE), [f"{i}" for i in range(*CONSTS_RANGE)])
    plt.ylabel("Tasks count", fontsize=12)
    plt.xlabel("Constraints per task", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, f"histogram_of_num_constraints{name_add}.png"))


def plot_frequency_of_constraints(all_constraints):
    count_freq = Counter(all_constraints)
    count_counts = Counter(count_freq.values())
    keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]
    values = []
    for i in range(1, 10):
        values.append(count_counts[i])
    values.append(sum([count_counts[i] for i in range(10, max(count_counts.keys()) + 1)]))
    plt.bar(np.arange(1, 11), values, color='#1f77b4', edgecolor='black')
    plt.yscale('log')
    plt.xticks(range(1, 11), keys)
    plt.xlabel("Number of occurrences of a constraint in the dataset", fontsize=12)
    plt.ylabel("Number of unique constraints (log scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "frequency_of_constraints.png"))


def get_all_scores(all_jsons, ds, domain_df):
    print("calculating scores...")
    all_scores = {k: [] for k in all_jsons}
    all_constraints_scores = {k: [] for k in all_jsons}
    df_categories = pd.read_csv(CONSTRAINT_CATEGORY_CLASSIFICATION)
    dict_constraints = {}
    all_constraints_from_ds = set()
    # for each constraint, map to a category
    for _, row in df_categories.iterrows():
        category = row["categories"]
        other = row["Other"]
        if other == 1 or pd.isna(category):
            category = "Other"
        dict_constraints[row["constraint"]] = category
    for k in tqdm(all_jsons):
        for i, row in enumerate(ds):
            task = row["task"]
            decomposition_len = len(row["decomposition"])
            num_constraints = len(all_jsons[k][task]["scores"])
            assert decomposition_len == num_constraints
            binary_scores = []
            for constraint in all_jsons[k][task]["scores"]:
                s = all_jsons[k][task]["scores"][constraint]
                if s == 'ERR':
                    s = 0
                binary_score_current_constraint = int(s >= 0.5)  # convert to binary
                pos_category = dict_constraints[constraint] # if constraint in dict_constraints else 'Other'
                all_constraints_from_ds.add(constraint)
                all_constraints_scores[k].append({"constraint": constraint, "category": pos_category,
                                                  "score": s, "binary_score": binary_score_current_constraint,
                                                  "total_constraints": num_constraints, "orig_task": task})
                binary_scores.append(binary_score_current_constraint)
            mean_score = np.mean(binary_scores)
            all_fulfilled = mean_score == 1
            all_scores[k].append((mean_score, all_fulfilled, num_constraints))
        all_scores[k] = pd.DataFrame(all_scores[k], columns=["mean_score", "all_fulfilled", "num_constraints"])
        all_constraints_scores[k] = pd.DataFrame(all_constraints_scores[k])
    dict_constraints = {constraint: dict_constraints[constraint] for constraint in all_constraints_from_ds}
    return all_scores, all_constraints_scores, dict_constraints


if __name__ == '__main__':
    llama_models = ["Llama-3.1-8B", "Llama-3.2-1B", "Llama-3.2-3B"]
    # llama_models = ["Llama-3.1-8B", "Llama-3.2-3B"]
    gemma_models = ["gemma-2-2b", "gemma-2-9b"]
    qwen_models = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]
    # qwen_models = ["Qwen2.5-3B", "Qwen2.5-7B"]
    big_models = ["llama3.3-70b", "mistral-large", "qwen2.5-72b", "llama3.1-405b", "deepseek-v3"]
    all_paths = {}

    for name, models in zip(["Llama", "Gemma", "Qwen", "Big models"], [llama_models, gemma_models, qwen_models, big_models]):
    # for name, models in zip(["Big models"], [big_models]):
        init = {model.capitalize(): PATH_TO_SCORES.format(model_name=model) for model in models}
        all_paths.update(init)

    calculate(all_paths)



