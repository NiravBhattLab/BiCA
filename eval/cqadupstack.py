from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import logging
import json
import pandas as pd
import os

def evaluate_dataset(dataset_path, dataset_name, batch_size=32, trust_remote_code=False):
    print(f"{'#' * 20} {dataset_name.upper()} {'#' * 20}")
    
    # load corpus
    corpus_path = os.path.join(dataset_path, "corpus.jsonl")
    with open(corpus_path, "r") as f:
        corpus = [json.loads(line) for line in f]

    # load queries
    queries_path = os.path.join(dataset_path, "queries.jsonl")
    with open(queries_path, "r") as f:
        queries = [json.loads(line) for line in f]

    queries = {query["_id"]: query["text"] for query in queries}
    corpus = {entry["_id"]: {"title": entry["title"], "text": entry["text"]} for entry in corpus}

    # load qrels
    qrels_path = os.path.join(dataset_path, "qrels", "test.tsv")
    qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "corpus_id", "score"])
    qrels["score"] = pd.to_numeric(qrels["score"], errors="coerce")
    qrels = qrels.dropna(subset=["score"])
    qrels["score"] = qrels["score"].astype(int)
    qrels = qrels.groupby('query_id', group_keys=False).apply(
        lambda group: {group['corpus_id'].iloc[i]: int(group['score'].iloc[i]) for i in range(len(group))}
    ).to_dict()

    gpl_name = "output/gtesmall"  # path to the trained model
    model = DRES(models.SentenceBERT(gpl_name, trust_remote_code=trust_remote_code, model_kwargs={'attn_implementation': 'eager'}), batch_size=batch_size)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    results = retriever.retrieve(corpus, queries)

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    print(f"NDCG@{retriever.k_values}: {ndcg}")
    print(f"MAP@{retriever.k_values}: {_map}")
    print(f"Recall@{retriever.k_values}: {recall}")
    print(f"Precision@{retriever.k_values}: {precision}")
    
    return ndcg.get("NDCG@10", 0.0)

cqadupstack_base_path = "eval_datasets/cqadupstack"
cqadupstack_datasets = [
    "android",
    "english", 
    "gaming",
    "gis",
    "mathematica",
    "physics",
    "programmers",
    "stats",
    "tex",
    "unix",
    "webmasters",
    "wordpress"
]

evaluation_results = {}
ndcg_scores = []

# evaluate each CQADupStack dataset
for dataset_name in cqadupstack_datasets:
    dataset_path = os.path.join(cqadupstack_base_path, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} does not exist. Skipping...")
        continue
    
    try:
        ndcg_10 = evaluate_dataset(dataset_path, dataset_name, batch_size=32, trust_remote_code=True)
        evaluation_results[f"cqadupstack-{dataset_name}"] = ndcg_10
        ndcg_scores.append(ndcg_10)
        print(f"NDCG@10 for {dataset_name}: {ndcg_10:.4f}")
        print("-" * 60)
    except Exception as e:
        print(f"Error evaluating {dataset_name}: {str(e)}")
        continue

# Calculate average NDCG@10 across all datasets
if ndcg_scores:
    average_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    evaluation_results["cqadupstack-average"] = average_ndcg
    
    print(f"{'=' * 60}")
    print(f"AVERAGE NDCG@10 across all CQADupStack datasets: {average_ndcg:.4f}")
    print(f"Evaluated {len(ndcg_scores)} datasets successfully")
    print(f"{'=' * 60}")
else:
    print("No datasets were successfully evaluated!")

# Save results to JSON file
with open("cqa-base.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)

print(f"Results saved to evaluation-cqa.json")