import logging
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

BASE_MODEL_NAME = "google-bert/bert-base-uncased"
ABLATION_MODELS_DIR = "output/bert"
EVAL_DATASETS_DIR = "eval_datasets"
RESULTS_FILE = "ablation_evaluation_results.json"
PLOT_FILE = "scaling_performance_plot_swapped_axes.png" 

DATASETS_CONFIG = [
    ("nfcorpus", 32, True),
    ("scidocs", 16, True),
    ("scifact", 16, False),
    ("arguana", 32, True),
]

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def evaluate_model_on_dataset(model_path, dataset_name, batch_size, trust_remote_code):
    """
    Evaluates a given model on a specified BEIR dataset.
    """
    print(f"\n{'='*20} EVALUATING {'='*20}")
    print(f"MODEL: {model_path}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*54}")

    corpus_path = os.path.join(EVAL_DATASETS_DIR, dataset_name, "corpus.jsonl")
    queries_path = os.path.join(EVAL_DATASETS_DIR, dataset_name, "queries.jsonl")
    qrels_path = os.path.join(EVAL_DATASETS_DIR, dataset_name, "qrels", "test.tsv")

    if not all(os.path.exists(p) for p in [corpus_path, queries_path, qrels_path]):
        logging.error(f"Dataset files for '{dataset_name}' not found. Skipping.")
        return 0.0

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_list = [json.loads(line) for line in f]
    with open(queries_path, "r", encoding="utf-8") as f:
        queries_list = [json.loads(line) for line in f]

    corpus = {entry["_id"]: {"title": entry.get("title", ""), "text": entry["text"]} for entry in corpus_list}
    queries = {query["_id"]: query["text"] for query in queries_list}

    qrels_df = pd.read_csv(qrels_path, sep="\t")
    qrels = {}
    for _, row in qrels_df.iterrows():
        if row['query-id'] not in qrels:
            qrels[str(row['query-id'])] = {}
        qrels[str(row['query-id'])][str(row['corpus-id'])] = int(row['score'])

    try:
        model = DRES(models.SentenceBERT(model_path, trust_remote_code=trust_remote_code, model_kwargs={'attn_implementation': 'eager'}), batch_size=batch_size)
    except Exception as e:
        logging.error(f"Failed to load model {model_path}. Error: {e}")
        return 0.0
        
    retriever = EvaluateRetrieval(model, score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)

    logging.info(f"Evaluating retriever for k in: {retriever.k_values}")
    ndcg, _, _, _ = retriever.evaluate(qrels, results, retriever.k_values)
    
    ndcg_10 = ndcg.get("NDCG@10", 0.0)
    print(f"NDCG@10 for {model_path} on {dataset_name}: {ndcg_10:.4f}")
    return ndcg_10

def generate_plot(results_df):
    """
    Generates and saves a high-quality plot from the evaluation results
    with axes swapped: X-axis = Datasets, Lines = Models.
    """
    logging.info(f"Generating performance plot and saving to {PLOT_FILE}...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for model_name in results_df.columns:
        ax.plot(
            results_df.index, 
            results_df[model_name], 
            marker='o',
            linestyle='-',
            label=model_name
        )

    ax.set_title("Model Performance Across Evaluation Datasets", fontsize=16, fontweight='bold')
    ax.set_xlabel("Evaluation Dataset", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_yscale('log')
    
    plt.xticks(rotation=10, ha='right')
    
    ax.legend(title="Training Corpus Size", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    
    fig.savefig(PLOT_FILE, dpi=600, bbox_inches='tight')
    logging.info(f"Plot successfully generated: {PLOT_FILE}")

def main():
    l =[1000,5000,10000,15000,'full']
    def sort_key(dir_name):
        if dir_name in l:
            return float('inf')
        try:
            return int(dir_name)
        except ValueError:
            return -1

    model_configs = [("Baseline", BASE_MODEL_NAME)]

    if os.path.isdir(ABLATION_MODELS_DIR):
        sub_dirs = sorted([d for d in os.listdir(ABLATION_MODELS_DIR) if os.path.isdir(os.path.join(ABLATION_MODELS_DIR, d))], key=sort_key)
        for dir_name in sub_dirs:
            model_path = os.path.join(ABLATION_MODELS_DIR, dir_name)
            if os.path.exists(model_path):
                model_configs.append((dir_name, model_path))
            else:
                 logging.warning(f"final_model not found in {os.path.join(ABLATION_MODELS_DIR, dir_name)}. Skipping.")
    
    if len(model_configs) <= 1:
        logging.error("No fine-tuned models found in the ablation directory. Exiting.")
        return

    all_results = {}
    for model_name, model_path in model_configs:
        for dataset_name, batch_size, trust_remote_code in DATASETS_CONFIG:
            if dataset_name not in all_results:
                all_results[dataset_name] = {}
            
            ndcg_10 = evaluate_model_on_dataset(model_path, dataset_name, batch_size, trust_remote_code)
            all_results[dataset_name][model_name] = ndcg_10

    if not all_results:
        logging.error("No evaluation results were generated. Cannot create plot.")
        return

    logging.info(f"Saving all evaluation results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=4)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.T # Transpose so datasets are rows and models are columns

    ordered_columns = [name for name, _ in model_configs]
    results_df = results_df[ordered_columns]
    
    print("\n--- Final Evaluation Results Summary ---")
    print(results_df)
    print("--------------------------------------")
    
    generate_plot(results_df)

if __name__ == "__main__":
    main()