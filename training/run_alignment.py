import json
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DAPT_MODEL_PATH = "../models/e5-yoga-neuro-dapt"
ALIGNMENT_FILE = "../data/processed/alignment_pairs.json"
OUTPUT_PATH = "../models/e5-yoga-neuro-aligned"

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5

# -------------------------------------------------
# Check Device
# -------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = SentenceTransformer(DAPT_MODEL_PATH, device=device)

# -------------------------------------------------
# Load Alignment Data
# -------------------------------------------------

with open(ALIGNMENT_FILE, "r", encoding="utf-8") as f:
    pairs = json.load(f)

train_examples = []

for pair in pairs:
    train_examples.append(
        InputExample(
            texts=[pair["anchor"], pair["positive"]]
        )
    )

print("Total alignment pairs:", len(train_examples))

# -------------------------------------------------
# DataLoader
# -------------------------------------------------

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE
)

# -------------------------------------------------
# Loss (Multiple Negatives Ranking)
# -------------------------------------------------

train_loss = losses.MultipleNegativesRankingLoss(model)

# -------------------------------------------------
# Training
# -------------------------------------------------

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=int(len(train_dataloader) * EPOCHS * 0.1),
    optimizer_params={"lr": LEARNING_RATE},
    show_progress_bar=True
)

# -------------------------------------------------
# Save Model
# -------------------------------------------------

model.save(OUTPUT_PATH)

print("Alignment training complete.")
print("Model saved to:", OUTPUT_PATH)
