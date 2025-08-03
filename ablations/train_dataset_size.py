import json
import os
import math
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers import models as sbert_models
from sentence_transformers.evaluation import TripletEvaluator

MODEL_NAME = 'google-bert/bert-base-uncased'
INPUT_JSONL_FILE = 'hard-negatives-traversal.jsonl'
BASE_OUTPUT_PATH = 'output/bert'

BATCH_SIZE = 16
NUM_EPOCHS = 1
VALIDATION_SPLIT_RATIO = 0.1
EVALUATION_STEPS = 50

ABLATION_SIZES = [1000, 5000, 10000, 15000, 'full']

print(f"Loading all training data from: {INPUT_JSONL_FILE}")
all_query_blocks = []
try:
    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                query = data.get('query')
                positive = data.get('positive')
                negatives = data.get('negatives', [])
                if not query or not positive or not negatives: continue
                all_query_blocks.append((query, positive, negatives))
            except (json.JSONDecodeError, Exception):
                continue
except FileNotFoundError:
    print(f"Error: Input file not found: {INPUT_JSONL_FILE}")
    exit()

if not all_query_blocks:
    print("No valid queries were loaded. Please check your JSONL file.")
    exit()

print(f"Loaded a total of {len(all_query_blocks)} queries.")
print("-" * 50)

random.shuffle(all_query_blocks)

for size in ABLATION_SIZES:
    current_size_str = str(size) if isinstance(size, int) else 'full'
    output_path = os.path.join(BASE_OUTPUT_PATH, current_size_str)
    
    print(f"\n--- Checking for Ablation size: {current_size_str} ---")

    if os.path.exists(os.path.join(output_path, 'pytorch_model.bin')):
        print(f"Skipping size '{current_size_str}': Best model already saved at '{output_path}'")
        continue

    print(f"Starting new training for size: {current_size_str}")
    print(f"Loading pre-trained model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    if isinstance(size, int):
        if size > len(all_query_blocks):
            print(f"Warning: Ablation size {size} is larger than the total dataset ({len(all_query_blocks)}). Skipping.")
            continue
        selected_query_blocks = all_query_blocks[:size]
    else:
        selected_query_blocks = all_query_blocks

    split_index = int(len(selected_query_blocks) * (1 - VALIDATION_SPLIT_RATIO))
    train_blocks = selected_query_blocks[:split_index]
    dev_blocks = selected_query_blocks[split_index:]
    
    print(f"Using {len(train_blocks)} queries for training and {len(dev_blocks)} for validation.")

    train_examples = []
    for query, positive, negatives in train_blocks:
        for negative_text in negatives:
            train_examples.append(InputExample(texts=[query, positive, negative_text]))

    if not train_examples:
        print("No training examples for this subset. Skipping.")
        continue

    dev_examples = []
    for query, positive, negatives in dev_blocks:
        dev_examples.append(InputExample(texts=[query, positive, negatives[0]]))

    print(f"Created {len(train_examples)} training triplets.")
    print(f"Created {len(dev_examples)} validation triplets.")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    dev_evaluator = TripletEvaluator.from_input_examples(dev_examples, name=f'dev-{current_size_str}')

    os.makedirs(output_path, exist_ok=True)
    print(f"Best model will be saved to: {output_path}")

    print("Starting model fine-tuning with early stopping...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=EVALUATION_STEPS,
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"--- Completed Ablation for size: {current_size_str} ---")

print("\nFull ablation study complete.")