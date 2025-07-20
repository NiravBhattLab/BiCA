import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
from sentence_transformers import models as sbert_models

MODEL_NAME = 'thenlper/gte-base'    
#MODEL_NAME = 'thenlper/gte-small' 
INPUT_JSONL_FILE = 'hard.jsonl'
OUTPUT_PATH = 'output/gte' 
#OUTPUT_PATH = 'output/gtesmall'
BATCH_SIZE = 16
EPOCHS = 1
STEPS_PER_EPOCH = 20
# WARMUP_STEPS = 250


print(f"Loading pre-trained model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, model_kwargs={'attn_implementation': 'eager'})

if not any(isinstance(module, sbert_models.Pooling) for module in model.modules()):
    word_embedding_model = model[0] # Assuming the first module is the transformer
    pooling_model = sbert_models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model.add_module("pooling", pooling_model)

train_examples = []
print(f"Loading training data from: {INPUT_JSONL_FILE}")
try:
    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                query = data['query']
                positive = data['positive']
                for negative in data['negatives']:
                    train_examples.append(InputExample(
                        texts=[query, positive, negative]
                    ))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line: {line.strip()}")
            except KeyError as e:
                print(f"Warning: Skipping line due to missing key {e}: {line.strip()}")
except FileNotFoundError:
    print(f"Error: Input file not found: {INPUT_JSONL_FILE}")
    print("Please ensure the file exists and the path is correct.")
    exit()


if not train_examples:
    print("No training examples were loaded. Please check your JSONL file.")
    exit()

print(f"Loaded {len(train_examples)} training examples.")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

train_loss = losses.MultipleNegativesRankingLoss(model=model)

os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"Model will be saved to: {OUTPUT_PATH}")

num_train_steps = len(train_dataloader)
if STEPS_PER_EPOCH is None or STEPS_PER_EPOCH > num_train_steps:
    print(f"Adjusting steps_per_epoch from {STEPS_PER_EPOCH} to {num_train_steps} (all available batches).")
    actual_steps_per_epoch = num_train_steps
else:
    actual_steps_per_epoch = STEPS_PER_EPOCH

total_training_steps = actual_steps_per_epoch * EPOCHS
print("Starting model fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    steps_per_epoch = actual_steps_per_epoch,
    # warmup_steps=WARMUP_STEPS,
    output_path=OUTPUT_PATH,
    show_progress_bar=True,
)

final_save_path = os.path.join(OUTPUT_PATH, "final_model")
model.save(final_save_path)
print(f"Model explicitly saved to: {final_save_path}")
print("Fine-tuning complete.")
