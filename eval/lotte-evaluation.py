import argparse
from collections import defaultdict
import jsonlines
import os
import sys
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def load_passages(passages_path):
    """Load passages from a TSV with pid \t text."""
    pid_to_text = {}
    with open(passages_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading passages", unit="passage"):
            pid, text = line.strip().split('\t', 1)
            pid_to_text[int(pid)] = text
    return pid_to_text


def load_qas(qas_path):
    """Reads qid, query, and answer_pids; no per-query candidates."""
    qas = []
    with jsonlines.open(qas_path, mode="r") as reader:
        # Convert to list to get length for tqdm
        data = list(reader)
        for obj in tqdm(data, desc="Loading Q&As", unit="qa"):
            qas.append({
                "qid": int(obj["qid"]),
                "query": obj.get("query", obj.get("question", "")),
                "answer_pids": set(obj.get("answer_pids", [])),
            })
    return qas


def encode_passages_in_batches(model, texts, batch_size=16):
    """Encode passages in batches to manage memory usage."""
    all_embeddings = []
    
    batch_iter = tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding passages",
        unit="batch",
        total=(len(texts) + batch_size - 1) // batch_size
    )
    
    for i in batch_iter:
        batch = texts[i:i + batch_size]
        batch_embs = model.encode(
            batch, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        all_embeddings.append(batch_embs)
        
        # Update progress bar description
        batch_iter.set_postfix({
            'passages': f"{min(i + batch_size, len(texts))}/{len(texts)}"
        })
    
    # Concatenate all embeddings
    return torch.cat(all_embeddings, dim=0)


def encode_queries_in_batches(model, queries, batch_size=16):
    """Encode queries in batches."""
    all_embeddings = []
    
    batch_iter = tqdm(
        range(0, len(queries), batch_size),
        desc="Encoding queries",
        unit="batch",
        total=(len(queries) + batch_size - 1) // batch_size
    )
    
    for i in batch_iter:
        batch = queries[i:i + batch_size]
        batch_embs = model.encode(
            batch, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        all_embeddings.append(batch_embs)
        
        # Update progress bar description
        batch_iter.set_postfix({
            'queries': f"{min(i + batch_size, len(queries))}/{len(queries)}"
        })
    
    return torch.cat(all_embeddings, dim=0)


def rerank(model, qas, pid_to_text, batch_size=16):
    """
    For each query, encode against *all* passages in pid_to_text.
    Pre-encode passages once for efficiency with batch processing.
    """
    all_pids = list(pid_to_text.keys())
    all_texts = [pid_to_text[pid] for pid in all_pids]
    
    # Encode all passages in batches
    passage_embs = encode_passages_in_batches(model, all_texts, batch_size)
    
    # Extract queries and encode them in batches
    queries = [qa["query"] for qa in qas]
    query_embs = encode_queries_in_batches(model, queries, batch_size)
    
    # Compute similarities and rank
    rankings = defaultdict(list)
    
    query_iter = tqdm(enumerate(qas), desc="Ranking queries", unit="query", total=len(qas))
    
    for i, qa in query_iter:
        qid = qa["qid"]
        q_emb = query_embs[i:i+1]  # Keep as 2D tensor
        scores = util.cos_sim(q_emb, passage_embs)[0].tolist()
        ranked = sorted(zip(all_pids, scores), key=lambda x: -x[1])
        for rank, (pid, _) in enumerate(ranked, start=1):
            rankings[qid].append((pid, rank))
        
        # Update progress bar
        query_iter.set_postfix({
            'qid': qid,
            'completed': f"{i + 1}/{len(qas)}"
        })
    
    return rankings


def save_rankings(rankings, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding='utf-8') as f:
        for qid, pid_rank_list in rankings.items():
            for pid, rank in pid_rank_list:
                f.write(f"{qid}\t{pid}\t{rank}\n")


def evaluate(qas, rankings, k):
    success = 0
    total = len(qas)
    for qa in qas:
        qid = qa["qid"]
        if qid not in rankings:
            print(f"WARNING: qid {qid} not in rankings!", file=sys.stderr)
            continue
        preds = [pid for pid, _ in rankings[qid][:k]]
        if set(preds).intersection(qa["answer_pids"]):
            success += 1
    return success / total * 100 if total > 0 else 0.0


def evaluate_dataset(model, query_type, dataset, split, k, data_rootdir, rankings_rootdir, batch_size=16):
    data_path = os.path.join(data_rootdir, dataset, split)
    qas_path = os.path.join(data_path, f"qas.{query_type}.jsonl")
    passages_path = os.path.join(data_path, "collection.tsv")
    rankings_path = os.path.join(rankings_rootdir, split, f"{dataset}.{query_type}.ranking.tsv")

    if not os.path.exists(qas_path) or not os.path.exists(passages_path):
        print(f"Skipping {dataset} ({query_type}) — files missing")
        return

    print(f"Processing [{query_type} | {dataset}] with batch_size={batch_size}...")
    pid_to_text = load_passages(passages_path)
    qas = load_qas(qas_path)

    rankings = rerank(model, qas, pid_to_text, batch_size)
    save_rankings(rankings, rankings_path)
    score = evaluate(qas, rankings, k)
    print(f"[query_type={query_type}, dataset={dataset}] Success@{k}: {score:.1f}%")


def main(args):
    model = SentenceTransformer(args.model_path)
    
    # Check if CUDA is available and inform user
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # List only the datasets you have; others will skip
    for query_type in ["forum"]:
        for dataset in [
            "writing",
            "recreation",
            #"science",
            #"technology",
            #"lifestyle",
            #"pooled",
        ]:
            evaluate_dataset(
                model,
                query_type,
                dataset,
                args.split,
                args.k,
                args.data_dir,
                args.rankings_dir,
                args.batch_size,
            )
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoTTE rerank + evaluate with custom SentenceTransformer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to custom SentenceTransformer model")
    parser.add_argument("--k", type=int, default=5, help="Success@k")
    parser.add_argument("-s", "--split", choices=["dev", "test"], required=True, help="Split to evaluate")
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to LoTTE data directory")
    parser.add_argument("-r", "--rankings_dir", type=str, required=True, help="Path to store output rankings")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding (default: 16)")
    args = parser.parse_args()
    main(args)