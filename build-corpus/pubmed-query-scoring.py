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
OUTPUT_FILE = "hard-negatives-traversal.jsonl" 
MODEL_NAME_QUERY = 'doc2query/all-t5-base-v1'
MODEL_NAME_SENTENCE = 'NeuML/pubmedbert-base-embeddings'

BATCH_SIZE = 128
NUM_TRAVERSAL_PATHS = 3
PATH_LENGTH = 3
TOP_K_SAMPLING = 5
INCLUDE_RANDOM_NEGATIVE = True

query_tokenizer = None
query_model = None
sentence_model = None
DEVICE = None

def setup_query_model():
    global query_tokenizer, query_model
    query_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME_QUERY)
    query_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME_QUERY)
    query_model.to(DEVICE)
    query_model.eval()

def setup_sentence_model():
    global sentence_model
    sentence_model = SentenceTransformer(MODEL_NAME_SENTENCE, device=DEVICE)

def setup_models():
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {DEVICE}")
    setup_query_model()
    setup_sentence_model()
    logging.info("Models setup complete.")

def generate_queries_batch(abstract_texts):
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
    if not sentence_model: return None
    return sentence_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

def build_dense_graph(query, hop1_abstracts, hop2_abstracts, positive_pmid):
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
    
    if not all_abstracts_to_embed:
        return None

    all_embeddings = get_embeddings(all_abstracts_to_embed)
    if all_embeddings is None:
        return None

    for i, meta in enumerate(doc_metadata):
        all_documents.append({
            'pmid': meta['pmid'], 
            'abstract': all_abstracts_to_embed[i], 
            'hop': meta['hop']
        })

    if len(all_documents) < 1:
        return None
    
    embeddings_tensor = all_embeddings
    hop1_indices = [i for i, doc in enumerate(all_documents) if doc['hop'] == 1]
    if not hop1_indices:
        return None
        
    hop1_embeddings = embeddings_tensor[hop1_indices]
    query_similarities = cosine_similarity(query_embedding.unsqueeze(0), hop1_embeddings).squeeze(0)
    
    normalized_embeddings = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
    pairwise_similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
    
    return {
        'query': query, 'documents': all_documents, 'embeddings': embeddings_tensor,
        'query_similarities': query_similarities, 'hop1_indices': hop1_indices,
        'pairwise_similarities': pairwise_similarities, 'positive_pmid': positive_pmid
    }

def find_diverse_hard_negatives(dense_graph):
    query_similarities = dense_graph['query_similarities']
    hop1_indices = dense_graph['hop1_indices']
    documents = dense_graph['documents']
    pairwise_similarities = dense_graph['pairwise_similarities']
    positive_pmid = dense_graph['positive_pmid']
    
    all_hard_negatives = set()
    globally_visited_nodes = set()

    num_starting_points = min(NUM_TRAVERSAL_PATHS, len(hop1_indices))
    if num_starting_points == 0: return []
    
    top_start_indices_in_hop1 = torch.topk(query_similarities, k=num_starting_points).indices
    
    if top_start_indices_in_hop1.dim() == 0:
        top_start_indices_in_hop1 = top_start_indices_in_hop1.unsqueeze(0)
    
    for start_idx_in_hop1 in top_start_indices_in_hop1:
        current_node_idx = hop1_indices[start_idx_in_hop1.item()]
        
        if current_node_idx in globally_visited_nodes:
            continue
        globally_visited_nodes.add(current_node_idx)

        start_doc = documents[current_node_idx]
        if start_doc['pmid'] != positive_pmid:
            all_hard_negatives.add(start_doc['abstract'])
        
        for _ in range(PATH_LENGTH - 1):
            current_similarities = pairwise_similarities[current_node_idx].clone()
            current_similarities[list(globally_visited_nodes)] = -1.0
            
            num_available = (current_similarities > -1.0).sum().item()
            if num_available == 0:
                break

            k_for_sampling = min(TOP_K_SAMPLING, num_available)
            
            top_k_values, top_k_indices = torch.topk(current_similarities, k=k_for_sampling)
            
            # Handle the edge case where topk returns a scalar for k=1
            if top_k_indices.dim() == 0:
                top_k_indices = top_k_indices.unsqueeze(0)
                
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

def process_batch(batch):
    positive_abstracts = [record['positive_abstract'] for record in batch]
    queries = generate_queries_batch(positive_abstracts)
    
    processed_records = []
    for record, query in zip(batch, queries):
        if not query:
            continue
        
        try:
            initial_pmid = record['initial_pmid']
            hop1_abstracts = record['hop1_abstracts']
            hop2_abstracts = record['hop2_abstracts']
            
            dense_graph = build_dense_graph(query, hop1_abstracts, hop2_abstracts, initial_pmid)
            if not dense_graph:
                continue
            
            hard_negatives = find_diverse_hard_negatives(dense_graph)
            if not hard_negatives:
                continue
                
            processed_records.append({"query": query, "positive": record['positive_abstract'], "negatives": hard_negatives})
        
        except Exception as e:
            logging.error(f"Error processing record {record.get('initial_pmid', 'unknown')}: {e}\n{traceback.format_exc()}")
            
    return processed_records

def load_2hop_records():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file {INPUT_FILE} not found.")
    records = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try: records.append(json.loads(line.strip()))
            except json.JSONDecodeError: continue
    return records

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting diverse hard negative generation...")
    setup_models()
    records = load_2hop_records()
    logging.info(f"Loaded {len(records)} 2-hop citation records.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Generating Diverse Hard Negatives"):
            batch = records[i:i+BATCH_SIZE]
            processed_batch = process_batch(batch)
            for final_record in processed_batch:
                if final_record:
                    f_out.write(json.dumps(final_record) + '\n')
    
    logging.info(f"Finished. Diverse hard negatives data saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
