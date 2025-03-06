import os.path
import numpy as np
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from data_analysis.utils import generate_color_map
from evalute_llms.utils import load_data
import gdown

GENERATIONS_DIR = "model_predictions"
ANALYSIS_DIR = "analysis_output"
CONSTRAINT_CATEGORY_CLASSIFICATION = os.path.join(ANALYSIS_DIR, "constraint_categories.csv")
DOMAINS_PATH = os.path.join(ANALYSIS_DIR, "domains.csv")
ATOMICS_EMBEDDINGS = os.path.join(ANALYSIS_DIR, "atomics.npy")
ATOMICS_SORTED = os.path.join(ANALYSIS_DIR, "sorted_atomics.json")


SCORES_DIR = "llm_aaj_scores"
SCORES_FNAME = "llm-aaj-llama3.1-70b.{model_name}-0shot-wild-if-eval.json"
PATH_TO_SCORES = os.path.join(SCORES_DIR, SCORES_FNAME)

OUT_DIR_FOR_PLOTS = os.path.join("analysis_output", "figures")


def sort_model_name(word):
    first_letter = word[0]  # First letter
    try:
        middle_letter = float(word.split('-')[-1][:-1])  # Middle letter
    except ValueError:
        middle_letter = 0
    return first_letter, middle_letter  # Sorting by first, then middle letter


def calculate(paths):

    color_map_by_model = generate_color_map()
    all_jsons = {}
    ds = load_data("gililior/wild-if-eval")

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

    categories_sorted = sorted(set(constraint_to_category.values()))
    colors_for_categories = {c: plt.cm.tab20.colors[i * 2] for i, c in enumerate(categories_sorted)}
    if "Other" not in categories_sorted:
        colors_for_categories["Other"] = plt.cm.tab20.colors[len(categories_sorted) * 2]

    plot_constraint_embeddings(all_unique_constraints, constraint_to_category, categories_sorted, colors_for_categories)

    models_sorted = sorted(all_scores.keys(), key=sort_model_name)
    models_sorted.reverse()

    model_colors = [color_map_by_model[model] for model in models_sorted]

    # count how many constraints are in each category
    count = Counter(constraint_to_category.values())
    category_constraint_labels = list(count.keys())
    category_constraint_labels = sorted(category_constraint_labels, key=lambda x: count[x], reverse=True)

    plot_category_frequency(all_constraints_scores, category_constraint_labels, colors_for_categories, count)

    category_constraint_labels = category_constraint_labels[:-1]  # remove "Other" category

    plot_co_occurrences(category_constraint_labels, constraint_to_category, domain_df, ds)

    means = plot_performance_as_function_of_num_constraints(all_constraints_scores, all_scores, model_colors,
                                                            models_sorted)

    plot_mean_scores(means, model_colors, models_sorted)

    print("\n\nMean scores:")
    for model in all_scores:
        print(f"{model}: {all_scores[model]['mean_score'].mean().round(3)}")

    plot_score_by_constraint_category(all_constraints_scores, category_constraint_labels,
                                      constraint_to_category, model_colors, models_sorted)



def plot_score_by_constraint_category(all_constraints_scores, category_constraint_labels, constraint_to_category,
                                      model_colors, models_sorted):
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
    models_to_include = ["Deepseek-v3", "Llama3.3-70b", "Mistral-large", "Llama3.1-405b", "Qwen2.5-72b", "Gemma-2-9b"]
    reorg_categories = ["Focus / Emphasis", "Style and Tone", "Persona and Role", "Include / Avoid", "Ensure Quality",
                        "Format and Structure", "Editing", "Length"]
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
    ax.set_ylabel('Fraction of fulfilled constraints')
    ax.legend(bbox_to_anchor=(0, 1.1), loc='upper left', ncol=len(models_to_include))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "bar_plot_of_mean_score_by_category_only_big_models.png"))


def plot_mean_scores(means, model_colors, models_sorted):
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
    plt.ylabel("Mean fraction of fulfilled constraints", fontsize=12)
    plt.ylim(0, 0.75)
    plt.xticks(rotation=45, ha='right')  # Rotate labels if needed
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "bar_plot_of_mean_score.png"))


def plot_performance_as_function_of_num_constraints(all_constraints_scores, all_scores, model_colors, models_sorted):
    plt.figure("line plot normalized")
    score_1 = {}
    means = []
    for i, model in enumerate(models_sorted):
        df = all_scores[model]
        all_means = all_constraints_scores[model].groupby(["total_constraints", "category"])["score"].mean()
        mean_score_by_num_constraint = []
        for num_constraints in range(1, 9):
            mean_data = all_means.get(num_constraints).mean()
            mean_score_by_num_constraint.append(mean_data)
        means.append(df["mean_score"].mean())
        score_1[model] = mean_score_by_num_constraint[0]
        # plot line plot
        plt.plot(range(1, 9), mean_score_by_num_constraint, label=model, color=model_colors[i])
    plt.xlabel("Number of constraints in a task", fontsize=12)
    plt.ylabel("Fraction of fulfilled constraints (normalized)", fontsize=12)
    plt.xticks(range(1, 9))
    # sort the order in legend
    handles, labels_for_figure = plt.gca().get_legend_handles_labels()
    custom_order = sorted(score_1.keys(), key=lambda x: -score_1[x])
    # Sort by custom order
    sorted_handles_labels = sorted(zip(handles, labels_for_figure), key=lambda x: custom_order.index(x[1]))
    # Unzip into sorted handles and labels
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    # Add legend
    plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
    # plt.title(f"score by num constraints")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "line_plot_of_mean_score_by_num_constraints_normalized.png"))
    return means


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
    for i in range(1, 9):
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


def plot_histogram_of_num_constraints_in_task(all_lengths):
    plt.figure("histogram of num constraints")
    plt.bar(range(1, 9), [all_lengths.count(i) for i in range(1, 9)], color='#ff7f0e', edgecolor='black')
    plt.xticks(range(1, 9), [f"{i}" for i in range(1, 9)])
    plt.ylabel("Tasks count", fontsize=12)
    plt.xlabel("Constraints per task", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FOR_PLOTS, "histogram_of_num_constraints.png"))


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
            if task not in all_jsons[k]:
                continue
            num_constraints = len(all_jsons[k][task]["scores"])
            domain = domain_df[domain_df["task"] == task]["domain"].values[0]
            if pd.isna(domain):
                domain = "Other"
            scores = []
            for constraint in all_jsons[k][task]["scores"]:
                s = all_jsons[k][task]["scores"][constraint]
                if s == 'ERR':
                    s = 0
                pos_category = dict_constraints[constraint] if constraint in dict_constraints else 'Other'
                all_constraints_scores[k].append({"constraint": constraint, "category": pos_category,
                                                  "score": s, "total_constraints": num_constraints, "orig_task": task})
                scores.append(s)
            binary_scores = [1 if s >= 0.5 else 0 for s in scores]
            if len(binary_scores) == 0:
                print(f"no scores for {task}")
                continue
            mean_score = np.mean(binary_scores)
            num_constraints = len(binary_scores)
            all_scores[k].append((mean_score, num_constraints, decomposition_len, domain))
        all_scores[k] = pd.DataFrame(all_scores[k], columns=["mean_score", "num_constraints", "decomposition_len", "domain"])
        all_constraints_scores[k] = pd.DataFrame(all_constraints_scores[k])
    return all_scores, all_constraints_scores, dict_constraints


if __name__ == '__main__':
    llama_models = ["Llama-3.1-8B", "Llama-3.2-1B", "Llama-3.2-3B"]
    gemma_models = ["gemma-2-2b", "gemma-2-9b"]
    qwen_models = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]
    big_models = ["llama3.3-70b", "mistral-large", "qwen2.5-72b", "llama3.1-405b", "deepseek-v3"]
    all_paths = {}

    for name, models in zip(["Llama", "Gemma", "Qwen", "Big models"], [llama_models, gemma_models, qwen_models, big_models]):
        init = {model.capitalize(): PATH_TO_SCORES.format(model_name=model) for model in models}
        all_paths.update(init)

    calculate(all_paths)



