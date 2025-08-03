import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
from sentence_transformers import models as sbert_models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import random
import logging
import glob

logging.basicConfig(level=logging.INFO)

MODEL_NAME = 'bert-base-uncased'
INPUT_FOLDER = 'hn_ablation_main'
BASE_OUTPUT_PATH = 'ablation_main'
BATCH_SIZE = 32
EPOCHS = 1
EVALUATION_STEPS = 50
EARLY_STOPPING_PATIENCE = 3

input_files = glob.glob(os.path.join(INPUT_FOLDER, '*.jsonl'))

if not input_files:
    print(f"Error: No .jsonl files found in the directory: {INPUT_FOLDER}")
    exit()

for input_file in input_files:
    file_name_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(BASE_OUTPUT_PATH, file_name_without_ext)

    print("-" * 80)
    print(f"Processing file: {input_file}")
    print(f"Loading pre-trained model: {MODEL_NAME}")

    word_embedding_model = sbert_models.Transformer(MODEL_NAME)
    pooling_model = sbert_models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    all_data = []
    print(f"Loading training data from: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line: {line[:100]}")
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        continue  

    if not all_data:
        print(f"No data loaded from {input_file}. Skipping.")
        continue

    random.seed(42)
    random.shuffle(all_data)

    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    dev_data = all_data[split_idx:]

    print(f"Data split for {os.path.basename(input_file)}: {len(train_data)} training examples, {len(dev_data)} development examples.")

    train_examples = []
    for item in train_data:
        if 'query' in item and 'positive' in item and 'negatives' in item:
            for neg in item['negatives']:
                train_examples.append(InputExample(texts=[item['query'], item['positive'], neg]))

    if not train_examples:
        print(f"No training examples were created from {input_file}. Please check that the JSONL file contains 'query', 'positive', and 'negatives' keys in each line. Skipping.")
        continue

    print(f"Created {len(train_examples)} training pairs.")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    dev_queries = {}
    dev_corpus = {}
    dev_relevant_docs = {}
    corpus_docs = set()
    doc_id_counter = 0

    for item in dev_data:
        if 'positive' in item:
            corpus_docs.add(item['positive'])
        if 'negatives' in item:
            for neg in item['negatives']:
                corpus_docs.add(neg)

    doc_to_id = {}
    for doc_text in corpus_docs:
        doc_id = f"doc_{doc_id_counter}"
        dev_corpus[doc_id] = doc_text
        doc_to_id[doc_text] = doc_id
        doc_id_counter += 1

    for i, item in enumerate(dev_data):
        query_id = f"q_{i}"
        dev_queries[query_id] = item['query']
        if 'positive' in item and item['positive'] in doc_to_id:
            positive_doc_id = doc_to_id[item['positive']]
            dev_relevant_docs[query_id] = {positive_doc_id}

    evaluator = InformationRetrievalEvaluator(
        queries=dev_queries,
        corpus=dev_corpus,
        relevant_docs=dev_relevant_docs,
        name='dev',
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )

    class EarlyStoppingException(Exception):
        pass

    class EarlyStopper:
        def __init__(self, patience: int = 3, min_delta: float = 0.0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_score = None

        def __call__(self, score, epoch, steps):
            if self.best_score is None:
                self.best_score = score
            elif score < self.best_score + self.min_delta:
                self.counter += 1
                logging.info(f"Early stopping counter: {self.counter}/{self.patience}. Current best score: {self.best_score:.4f}")
                if self.counter >= self.patience:
                    logging.info("Early stopping triggered.")
                    raise EarlyStoppingException()
            else:
                self.best_score = score
                self.counter = 0

    early_stopper = EarlyStopper(patience=EARLY_STOPPING_PATIENCE)

    os.makedirs(output_path, exist_ok=True)
    print(f"Model for {os.path.basename(input_file)} will be saved to: {output_path}")
    print(f"Starting model fine-tuning for {os.path.basename(input_file)}...")

    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=EPOCHS,
            evaluation_steps=EVALUATION_STEPS,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True,
            callback=early_stopper,
        )
    except EarlyStoppingException:
        print(f"Training for {os.path.basename(input_file)} stopped early due to validation performance.")

    print(f"Fine-tuning for {os.path.basename(input_file)} complete.")
    print(f"Best model for {os.path.basename(input_file)} saved in {output_path}")

print("-" * 80)
print("All files processed.")