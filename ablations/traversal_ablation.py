import json
import os
import torch
import logging
import traceback
import random
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

INPUT_FILE = "2hop-citation-graphs.jsonl"
OUTPUT_DIR = "hn_ablation_main"
MODEL_NAME_QUERY = 'doc2query/all-t5-base-v1'
MODEL_NAME_SENTENCE = 'NeuML/pubmedbert-base-embeddings'

BATCH_SIZE = 128
TOP_K_SAMPLING = 5
INCLUDE_RANDOM_NEGATIVE = True

DEFAULT_NUM_TRAVERSAL_PATHS = 3
DEFAULT_PATH_LENGTH = 3

TRAVERSAL_PATHS_RANGE = range(1, 6)  
PATH_LENGTH_RANGE = range(1, 6)      

query_tokenizer = None
query_model = None
sentence_model = None
DEVICE = None

def setup_query_model():
    """Initializes and loads the T5 query generation model."""
    global query_tokenizer, query_model
    query_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME_QUERY)
    query_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME_QUERY)
    query_model.to(DEVICE)
    query_model.eval()

def setup_sentence_model():
    """Initializes and loads the SentenceTransformer model for embeddings."""
    global sentence_model
    sentence_model = SentenceTransformer(MODEL_NAME_SENTENCE, device=DEVICE)

def setup_models():
    """Sets up the device (CPU/GPU) and loads all required models."""
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {DEVICE}")
    setup_query_model()
    setup_sentence_model()
    logging.info("Models setup complete.")

def generate_queries_batch(abstract_texts):
    """Generates a query for each abstract in a batch."""
    if not query_tokenizer or not query_model: return [None] * len(abstract_texts)
    input_texts = [f"generate query: {text}" for text in abstract_texts]
    inputs = query_tokenizer(input_texts, return_tensors="pt", max_length=512, truncation=True, padding=True).to(DEVICE)
    try:
        with torch.no_grad():
            outputs = query_model.generate(inputs.input_ids, max_length=64, num_beams=4, early_stopping=True)
        return query_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during query generation: {e}")
        return [None] * len(abstract_texts)

def get_embeddings(texts):
    """Computes embeddings for a list of texts."""
    if not sentence_model: return None
    return sentence_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

def build_dense_graph(query, hop1_abstracts, hop2_abstracts, positive_pmid):
    """Constructs a graph of documents with their embeddings and similarity matrices."""
    query_embedding = get_embeddings([query])
    if query_embedding is None: return None
    query_embedding = query_embedding[0]
    
    all_documents = []
    all_abstracts_to_embed = []
    doc_metadata = []

    for doc in hop1_abstracts:
        all_abstracts_to_embed.append(doc['abstract'])
        doc_metadata.append({'pmid': doc['pmid'], 'hop': 1})
    
    for doc in hop2_abstracts:
        all_abstracts_to_embed.append(doc['abstract'])
        doc_metadata.append({'pmid': doc['pmid'], 'hop': 2})
    
    if not all_abstracts_to_embed: return None
    all_embeddings = get_embeddings(all_abstracts_to_embed)
    if all_embeddings is None: return None

    for i, meta in enumerate(doc_metadata):
        all_documents.append({'pmid': meta['pmid'], 'abstract': all_abstracts_to_embed[i], 'hop': meta['hop']})

    if len(all_documents) < 1: return None
    
    embeddings_tensor = all_embeddings
    hop1_indices = [i for i, doc in enumerate(all_documents) if doc['hop'] == 1]
    if not hop1_indices: return None
        
    hop1_embeddings = embeddings_tensor[hop1_indices]
    query_similarities = cosine_similarity(query_embedding.unsqueeze(0), hop1_embeddings).squeeze(0)
    
    normalized_embeddings = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
    pairwise_similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    return {
        'query': query, 'documents': all_documents, 'embeddings': embeddings_tensor,
        'query_similarities': query_similarities, 'hop1_indices': hop1_indices,
        'pairwise_similarities': pairwise_similarities, 'positive_pmid': positive_pmid
    }

def find_diverse_hard_negatives(dense_graph, num_traversal_paths, path_length):
    """
    Finds hard negatives by performing traversals on the dense graph.
    This function now takes the ablation parameters as arguments.
    """
    query_similarities = dense_graph['query_similarities']
    hop1_indices = dense_graph['hop1_indices']
    documents = dense_graph['documents']
    pairwise_similarities = dense_graph['pairwise_similarities']
    positive_pmid = dense_graph['positive_pmid']
    
    all_hard_negatives = set()
    globally_visited_nodes = set()

    num_starting_points = min(num_traversal_paths, len(hop1_indices))
    if num_starting_points == 0: return []
    
    top_start_indices_in_hop1 = torch.topk(query_similarities, k=num_starting_points).indices
    
    if top_start_indices_in_hop1.dim() == 0:
        top_start_indices_in_hop1 = top_start_indices_in_hop1.unsqueeze(0)
    
    for start_idx_in_hop1 in top_start_indices_in_hop1:
        current_node_idx = hop1_indices[start_idx_in_hop1.item()]
        if current_node_idx in globally_visited_nodes: continue
        globally_visited_nodes.add(current_node_idx)

        start_doc = documents[current_node_idx]
        if start_doc['pmid'] != positive_pmid:
            all_hard_negatives.add(start_doc['abstract'])
        
        for _ in range(path_length - 1):
            current_similarities = pairwise_similarities[current_node_idx].clone()
            current_similarities[list(globally_visited_nodes)] = -1.0
            
            num_available = (current_similarities > -1.0).sum().item()
            if num_available == 0: break

            k_for_sampling = min(TOP_K_SAMPLING, num_available)
            top_k_values, top_k_indices = torch.topk(current_similarities, k=k_for_sampling)
            
            if top_k_indices.dim() == 0: top_k_indices = top_k_indices.unsqueeze(0)
                
            probs = torch.nn.functional.softmax(top_k_values, dim=0)
            next_node_choice_in_k = torch.multinomial(probs, 1).item()
            current_node_idx = top_k_indices[next_node_choice_in_k].item()
            
            globally_visited_nodes.add(current_node_idx)
            
            current_doc = documents[current_node_idx]
            if current_doc['pmid'] != positive_pmid:
                all_hard_negatives.add(current_doc['abstract'])

    hard_negative_list = list(all_hard_negatives)
    
    if INCLUDE_RANDOM_NEGATIVE:
        candidate_pool = [
            doc['abstract'] for i, doc in enumerate(documents) 
            if doc['pmid'] != positive_pmid and i not in globally_visited_nodes
        ]
        if candidate_pool:
            hard_negative_list.append(random.choice(candidate_pool))
            
    return hard_negative_list

def process_batch(batch, num_traversal_paths, path_length):
    """Processes a single batch of records with the given ablation parameters."""
    positive_abstracts = [record['positive_abstract'] for record in batch]
    queries = generate_queries_batch(positive_abstracts)
    
    processed_records = []
    for record, query in zip(batch, queries):
        if not query: continue
        
        try:
            dense_graph = build_dense_graph(query, record['hop1_abstracts'], record['hop2_abstracts'], record['initial_pmid'])
            if not dense_graph: continue
            
            hard_negatives = find_diverse_hard_negatives(dense_graph, num_traversal_paths, path_length)
            if not hard_negatives: continue
                
            processed_records.append({"query": query, "positive": record['positive_abstract'], "negatives": hard_negatives})
        
        except Exception as e:
            logging.error(f"Error processing record {record.get('initial_pmid', 'unknown')}: {e}\n{traceback.format_exc()}")
            
    return processed_records

def load_2hop_records():
    """Loads the input JSONL file."""
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file {INPUT_FILE} not found. Please place it in the same directory.")
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try: records.append(json.loads(line.strip()))
            except json.JSONDecodeError: continue
    return records

def run_pipeline_for_params(records, num_paths, path_len, run_num, total_runs):
    """
    A helper function to run the full generation pipeline for a given set of parameters.
    """
    logging.info(f"--- Starting Ablation Run {run_num}/{total_runs} ---")
    logging.info(f"Parameters: NUM_TRAVERSAL_PATHS = {num_paths}, PATH_LENGTH = {path_len}")

    output_filename = f"hn_traversals-{num_paths}_pathlen-{path_len}.jsonl"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_filepath, 'w', encoding='utf-8') as f_out:
        progress_desc = f"Run {run_num}/{total_runs} (Paths={num_paths}, Len={path_len})"
        for i in tqdm(range(0, len(records), BATCH_SIZE), desc=progress_desc):
            batch = records[i:i+BATCH_SIZE]
            processed_batch = process_batch(batch, num_paths, path_len)
            for final_record in processed_batch:
                if final_record:
                    f_out.write(json.dumps(final_record) + '\n')
    
    logging.info(f"Finished run. Data saved in {output_filepath}")

def main():
    """
    Main function to run the focused ablation study.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Ablation output will be saved in: {OUTPUT_DIR}")

    setup_models()
    records = load_2hop_records()
    logging.info(f"Loaded {len(records)} records for processing.")

    total_runs = len(PATH_LENGTH_RANGE) + len(TRAVERSAL_PATHS_RANGE) - 1
    current_run = 0

    logging.info(f"\n===== PART 1: Ablating PATH_LENGTH (NUM_TRAVERSAL_PATHS fixed at {DEFAULT_NUM_TRAVERSAL_PATHS}) =====")
    for path_len_variable in PATH_LENGTH_RANGE:
        current_run += 1
        run_pipeline_for_params(records, DEFAULT_NUM_TRAVERSAL_PATHS, path_len_variable, current_run, total_runs)

    logging.info(f"\n===== PART 2: Ablating NUM_TRAVERSAL_PATHS (PATH_LENGTH fixed at {DEFAULT_PATH_LENGTH}) =====")
    for num_paths_variable in TRAVERSAL_PATHS_RANGE:
        if num_paths_variable == DEFAULT_NUM_TRAVERSAL_PATHS:
            logging.info(f"Skipping run with NUM_TRAVERSAL_PATHS = {num_paths_variable} as it was completed in Part 1.")
            continue
        
        current_run += 1
        run_pipeline_for_params(records, num_paths_variable, DEFAULT_PATH_LENGTH, current_run, total_runs)

    logging.info("\nDone")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file '{INPUT_FILE}' not found.")
        print("Please make sure the file is in the same directory as the script.")
    else:
        main()