import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "intfloat/e5-base-v2"
DATA_FILE = "../data/processed/final_dapt_corpus.json"
OUTPUT_DIR = "../models/e5-yoga-neuro-dapt"

MAX_LENGTH = 256

# -----------------------------------
# Load Data
# -----------------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# -----------------------------------
# Load Model + Tokenizer
# -----------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

# -----------------------------------
# Tokenization
# -----------------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "domain"]
)

# -----------------------------------
# Data Collator (MLM)
# -----------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# -----------------------------------
# Training Arguments
# -----------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,
    save_total_limit=2,
    report_to="none"
)

# -----------------------------------
# Trainer
# -----------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("DAPT complete. Model saved to:", OUTPUT_DIR)
