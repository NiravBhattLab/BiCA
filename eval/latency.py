import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import time
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


model_path = "output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

dataset = load_dataset("Tevatron/msmarco-passage", split="train")

def encode_passages(dataset):
    all_embeddings = []
    num_passages = 0
    batch_size = 1000  # Reduce batch size if needed
    progress_bar = tqdm(desc="Encoding passages", unit=" passages")

    def process_batch(batch):
        passages = [example["query"] for example in batch]
        inputs = tokenizer(passages, padding=True, truncation=True, return_tensors="pt", max_length=512)

        # Move inputs to the same device as model
        inputs = {key: val.to("cuda") for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu()  # Move back to CPU

    current_batch = []
    for example in dataset:
        current_batch.append(example)
        num_passages += 1
        
        if len(current_batch) == batch_size:
            batch_embeddings = process_batch(current_batch)
            all_embeddings.append(batch_embeddings)
            current_batch = []
            progress_bar.update(batch_size)
        
        if num_passages >= 10000: 
            break

    if current_batch:
        batch_embeddings = process_batch(current_batch)
        all_embeddings.append(batch_embeddings)
        progress_bar.update(len(current_batch))

    progress_bar.close()
    print(f"Total passages encoded: {num_passages}")
    return torch.cat(all_embeddings)

# Ensure model is on CUDA
model.to("cuda")

# Encode passages
passage_embeddings = encode_passages(dataset)

# Create Faiss index
print("Creating Faiss index...")
dimension = passage_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(passage_embeddings.numpy())

# Function to measure latency
def measure_latency(queries, batch_size):
    # Measure encoding time
    start_time = time.time()
    inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt", max_length=512).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    query_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
    encoding_time = time.time() - start_time

    # Measure retrieval time (top-1000 passages)
    start_time = time.time()
    index.search(query_embeddings.numpy(), k=1000)
    retrieval_time = time.time() - start_time

    total_time = encoding_time + retrieval_time
    return encoding_time, retrieval_time, total_time

# Measure latency for different batch sizes
batch_sizes = [1, 10, 2000]
results = []

for batch_size in batch_sizes:
    print(f"Testing batch size {batch_size}...")
    
    # Fetch queries from the dataset
    queries = []
    for example in dataset:
        queries.append(example["query"])
        if len(queries) >= batch_size:
            break
    
    encoding_times = []
    retrieval_times = []
    total_times = []
    
    # Adjust number of iterations based on batch size
    iterations = 100 if batch_size < 2000 else 10
    
    for _ in range(iterations):
        encoding_time, retrieval_time, total_time = measure_latency(queries, batch_size)
        encoding_times.append(encoding_time)
        retrieval_times.append(retrieval_time)
        total_times.append(total_time)
    
    results.append({
        "Batch Size": batch_size,
        "Encoding Avg": np.mean(encoding_times) * 1000,
        "Encoding 99th": np.percentile(encoding_times, 99) * 1000,
        "Retrieval Avg": np.mean(retrieval_times) * 1000,
        "Retrieval 99th": np.percentile(retrieval_times, 99) * 1000,
        "Total Avg": np.mean(total_times) * 1000,
        "Total 99th": np.percentile(total_times, 99) * 1000
    })

# Print results
print("\nLatency analysis (times in milliseconds):")
print("Batch Size | Encoding (Avg/99th) | Retrieval (Avg/99th) | Total (Avg/99th)")
for result in results:
    print(f"{result['Batch Size']:10d} | {result['Encoding Avg']:.2f} / {result['Encoding 99th']:.2f} | {result['Retrieval Avg']:.2f} / {result['Retrieval 99th']:.2f} | {result['Total Avg']:.2f} / {result['Total 99th']:.2f}")