from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import logging
import json
import pandas as pd

def evaluate_dataset(dataset_name, batch_size=32, trust_remote_code=False):
    print(f"{'#' * 37}{dataset_name.upper()}{'#' * 37}")
    
    with open(f"eval/{dataset_name}/corpus.jsonl", "r") as f:
        corpus = [json.loads(line) for line in f]

    with open(f"eval/{dataset_name}/queries.jsonl", "r") as f:
        queries = [json.loads(line) for line in f]

    queries = {query["_id"]: query["text"] for query in queries}
    corpus = {entry["_id"]: {"title": entry["title"], "text": entry["text"]} for entry in corpus}

    qrels = pd.read_csv(f"eval/{dataset_name}/qrels/test.tsv", sep="\t", names=["query_id", "corpus_id", "score"])
    qrels["score"] = pd.to_numeric(qrels["score"], errors="coerce")
    qrels = qrels.dropna(subset=["score"])
    qrels["score"] = qrels["score"].astype(int)
    qrels = qrels.groupby('query_id', group_keys=False).apply(
        lambda group: {group['corpus_id'].iloc[i]: int(group['score'].iloc[i]) for i in range(len(group))}
    ).to_dict()

    gpl_name = "output"  # path to the trained model
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

evaluation_results = {}

datasets_config = [
    ("nfcorpus", 32, True),
    ("scidocs", 16, True),
    ("scifact", 16, False),
    ("trec-covid", 16, False),
    # ("nq", 128, False),
    # ("climate-fever", 128, False),
    # ("arguana", 32, True),
    # ("quora", 128, False)
]

for dataset_name, batch_size, trust_remote_code in datasets_config:
    ndcg_10 = evaluate_dataset(dataset_name, batch_size, trust_remote_code)
    evaluation_results[dataset_name] = ndcg_10

with open("evaluation-dsitill.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)