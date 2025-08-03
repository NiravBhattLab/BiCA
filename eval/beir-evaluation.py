from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import logging
import json
import os

def evaluate_dataset(dataset_name, batch_size=32, trust_remote_code=False):
    """
    Loads a local BEIR dataset, evaluates a model on it, and returns the nDCG@10 score.
    """
    print(f"{'#' * 37}{dataset_name.upper()}{'#' * 37}")

    folder_name = dataset_name.split('/')[0]
    data_path = os.path.join("eval_datasets", folder_name)
    
    if not os.path.exists(data_path):
        logging.error(f"Dataset not found at path: {data_path}")
        print(f"Dataset not found at path: {data_path}")
        return 0.0                 

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    model_name = "output/gte"
    model = DRES(models.SentenceBERT(model_name, trust_remote_code=trust_remote_code, model_kwargs={'attn_implementation': 'eager'}), batch_size=batch_size)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    results = retriever.retrieve(corpus, queries)

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    print(f"NDCG@{retriever.k_values}: {ndcg}")
    print(f"MAP@{retriever.k_values}: {_map}")
    print(f"Recall@{retriever.k_values}: {recall}")
    print(f"Precision@{retriever.k_values}: {precision}")
    
    return ndcg.get("NDCG@10", 0.0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    evaluation_results = {}

    datasets_config = [
        ("fever", 32, False),
        ("msmarco", 32, False),
        ('hotpotqa', 32, False),
        ("nfcorpus", 128, True),
        ("scidocs", 128, True),
        ("scifact", 128, False),
        ("trec-covid", 128, False),
        ("nq", 128, False),
        ("climate-fever", 128, False),
        ("arguana", 128, True),
        ("quora", 128, False),
        ("dbpedia-entity/dbpedia-entity", 32, False),
        ("fiqa", 32, False),
        ("webis-touche2020/webis-touche2020", 32, False)
    ]

    for dataset_name, batch_size, trust_remote_code in datasets_config:
        ndcg_10 = evaluate_dataset(dataset_name, batch_size, trust_remote_code)
        evaluation_results[dataset_name] = ndcg_10

    output_file = "evaluation-small.json"
    with open(output_file, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"\nEvaluation results saved to {output_file}")
    print(json.dumps(evaluation_results, indent=2))
