import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
from sentence_transformers import models as sbert_models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import random
import logging

logging.basicConfig(level=logging.INFO)

MODEL_NAME = 'sentence-transformers/gtr-t5-base'    
INPUT_JSONL_FILE = 'hard-negatives-traversal.jsonl'
OUTPUT_PATH = 'output/gtr' 
BATCH_SIZE = 16
EPOCHS = 1
EVALUATION_STEPS = 10
EARLY_STOPPING_PATIENCE = 3

print(f"Loading pre-trained model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, model_kwargs={'attn_implementation': 'eager'})

if not any(isinstance(module, sbert_models.Pooling) for module in model.modules()):
    word_embedding_model = model[0] 
    pooling_model = sbert_models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model.add_module("pooling", pooling_model)

all_data = []
print(f"Loading training data from: {INPUT_JSONL_FILE}")
try:
    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line: {line[:100]}")
except FileNotFoundError:
    print(f"Error: Input file not found: {INPUT_JSONL_FILE}")
    exit()

if not all_data:
    print("No data loaded. Exiting.")
    exit()

random.seed(42)
random.shuffle(all_data)

split_idx = int(len(all_data) * 0.8)
train_data = all_data[:split_idx]
dev_data = all_data[split_idx:]

print(f"Data split: {len(train_data)} training examples, {len(dev_data)} development examples.")

train_examples = []
for item in train_data:
    for neg in item.get('negatives', []):
        train_examples.append(InputExample(texts=[item['query'], item['positive'], neg]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

dev_queries = {}
dev_corpus = {}
dev_relevant_docs = {}
corpus_docs = set()
doc_id_counter = 0

for item in dev_data:
    corpus_docs.add(item['positive'])
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
    
    positive_doc_text = item['positive']
    positive_doc_id = doc_to_id[positive_doc_text]
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

fit_callback = None
if 't5' not in MODEL_NAME.lower():
    fit_callback = early_stopper
    logging.info("Enabled early stopping callback for non-T5 model.")
else:
    logging.info("Disabled early stopping callback for T5 model due to library incompatibility.")


os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"Model will be saved to: {OUTPUT_PATH}")
print("Starting model fine-tuning...")

try:
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        evaluation_steps=EVALUATION_STEPS,
        output_path=OUTPUT_PATH,
        save_best_model=True,
        show_progress_bar=True,
        callback=fit_callback,
    )
except EarlyStoppingException:
    print("Training stopped early due to validation performance.")

print("Fine-tuning complete.")
print(f"Best model saved in {OUTPUT_PATH}") 