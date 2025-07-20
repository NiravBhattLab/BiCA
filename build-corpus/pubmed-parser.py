import pubmed_parser as pp
from datasets import load_dataset
import json
import time
import os
from Bio import Entrez
import logging
import traceback
from tqdm import tqdm
import multiprocessing as mp

DATASET_NAME = "uiyunkim-hub/pubmed-abstract"
OUTPUT_FILE = "2hop-citation-graphs.jsonl"
TARGET_ITEMS_COUNT = 20000
HOPS_DESIRED = 2
DELAY_BETWEEN_API_CALLS = 0
DELAY_BETWEEN_TOP_LEVEL_PMIDS = 0
NCBI_EMAIL = "aarush.sinha@gmail.com"
NUM_WORKERS = 80

def setup_worker():
    """Initializer for each worker process."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
    Entrez.email = NCBI_EMAIL
    logging.info("Worker setup complete.")

def fetch_abstract_via_pubmed_parser(pmid_str):
    if DELAY_BETWEEN_API_CALLS > 0: time.sleep(DELAY_BETWEEN_API_CALLS)
    try:
        parsed_data = pp.parse_xml_web(pmid_str, save_xml=False)
        return parsed_data.get('abstract') if isinstance(parsed_data, dict) else None
    except Exception:
        return None

def fetch_outgoing_citations(pmid_str):
    if DELAY_BETWEEN_API_CALLS > 0: time.sleep(DELAY_BETWEEN_API_CALLS)
    try:
        return pp.parse_outgoing_citation_web(pmid_str, id_type='PMID')
    except Exception:
        return None

def build_2_hop_citation_graph(start_pmid_str, visited_pmids_in_path):
    """Build citation graph with separate hop1 and hop2 abstract arrays."""
    first_hop_citations = fetch_outgoing_citations(start_pmid_str)
    if not first_hop_citations or not first_hop_citations.get('pmid_cited'):
        return None
    
    hop1_abstracts = []
    hop2_abstracts = []
    
    for first_hop_pmid in first_hop_citations['pmid_cited']:
        if first_hop_pmid in visited_pmids_in_path: 
            continue
            
        first_hop_abstract = fetch_abstract_via_pubmed_parser(first_hop_pmid)
        if not first_hop_abstract: 
            continue
            
        # add to hop1 abstracts
        hop1_abstracts.append({
            "pmid": first_hop_pmid,
            "abstract": first_hop_abstract
        })
        
        # get second hop citations
        second_hop_citations_data = fetch_outgoing_citations(first_hop_pmid)
        
        if second_hop_citations_data and second_hop_citations_data.get('pmid_cited'):
            current_path_visited = visited_pmids_in_path | {first_hop_pmid}
            for second_hop_pmid in second_hop_citations_data['pmid_cited']:
                if second_hop_pmid in current_path_visited: 
                    continue
                    
                second_hop_abstract = fetch_abstract_via_pubmed_parser(second_hop_pmid)
                if second_hop_abstract:
                    hop2_abstracts.append({
                        "pmid": second_hop_pmid,
                        "abstract": second_hop_abstract
                    })
    
    if hop1_abstracts and hop2_abstracts:
        return {
            "hop1_abstracts": hop1_abstracts,
            "hop2_abstracts": hop2_abstracts
        }
    else:
        return None

def process_record(record):
    """Process a single record to build citation graph."""
    try:
        initial_pmid_str = str(record.get('pmid'))
        positive_abstract = record.get('abstract')
        
        if not initial_pmid_str or not positive_abstract or not positive_abstract.strip(): 
            return None
            
        citation_graph = build_2_hop_citation_graph(initial_pmid_str, {initial_pmid_str})
        if not citation_graph: 
            return None
            
        item_data = {
            "initial_pmid": initial_pmid_str, 
            "positive_abstract": positive_abstract, 
            "hop1_abstracts": citation_graph["hop1_abstracts"],
            "hop2_abstracts": citation_graph["hop2_abstracts"]
        }
        
        if DELAY_BETWEEN_TOP_LEVEL_PMIDS > 0: 
            time.sleep(DELAY_BETWEEN_TOP_LEVEL_PMIDS)
            
        return item_data
        
    except Exception as e:
        logging.error(f"Error processing PMID {record.get('pmid')}: {e}\n{traceback.format_exc()}")
        return None

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')
    logging.info(f"Starting citation graph parsing with {NUM_WORKERS} workers for {TARGET_ITEMS_COUNT} starting PMIDs (with 2-hop citation graphs).")
    
    processed_starting_pmids_count = 0
    already_processed_initial_pmids = set()
    
    # check for existing output file
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        already_processed_initial_pmids.add(json.loads(line)['initial_pmid'])
                    except (json.JSONDecodeError, KeyError):
                        logging.warning(f"Skipping malformed line in {OUTPUT_FILE}: {line.strip()}")
            processed_starting_pmids_count = len(already_processed_initial_pmids)
            logging.info(f"Resuming. Already processed {processed_starting_pmids_count} starting PMIDs.")
        except Exception as e:
            logging.error(f"Error reading {OUTPUT_FILE}: {e}")
    
    if processed_starting_pmids_count >= TARGET_ITEMS_COUNT:
        logging.info("Target starting PMID count already met.")
        return
    
    # load dataset
    logging.info("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split='train')
    logging.info(f"Dataset loaded with {len(dataset)} records.")
    
    def record_generator():
        """Generator to yield records in reverse order that haven't been processed."""
        for i in range(len(dataset) - 1, -1, -1):
            record = dataset[i]
            pmid_str = str(record.get('pmid'))
            if pmid_str not in already_processed_initial_pmids:
                yield record
    
    logging.info(f"Processing PMIDs from dataset in reverse order (starting from last record) until we get {TARGET_ITEMS_COUNT} successful citation graphs.")
    
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        logging.warning("Start method already set. Continuing.")
    
    pool = None
    successful_writes = 0
    total_attempts = 0
    
    try:
        pool = mp.Pool(processes=NUM_WORKERS, initializer=setup_worker)
        with tqdm(initial=processed_starting_pmids_count, total=TARGET_ITEMS_COUNT, desc="Successful Citations Written", unit="written") as pbar:
            for item_data in pool.imap_unordered(process_record, record_generator()):
                total_attempts += 1
                
                if item_data:
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                        f_out.write(json.dumps(item_data) + '\n')
                    successful_writes += 1
                    pbar.update(1)
                    
                if total_attempts % 1000 == 0:
                    success_rate = (successful_writes / total_attempts) * 100
                    logging.info(f"Progress: {total_attempts} PMIDs attempted, {successful_writes} citation graphs written ({success_rate:.1f}% success rate)")
                
                # break if we've reached our target of successful writes
                if successful_writes >= TARGET_ITEMS_COUNT:
                    break
        
        pool.close()
        pool.join()
        
        success_rate = (successful_writes / total_attempts) * 100 if total_attempts > 0 else 0
        logging.info(f"Final stats: {total_attempts} PMIDs attempted, {successful_writes} citation graphs written ({success_rate:.1f}% success rate)")
        
    except KeyboardInterrupt:
        logging.info("\nProcess interrupted. Terminating workers.")
        if pool: 
            pool.terminate()
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {e}\n{traceback.format_exc()}")
        if pool: 
            pool.terminate()
    finally:
        if pool: 
            pool.join()
        logging.info(f"Finished. Raw citation graphs saved in {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 