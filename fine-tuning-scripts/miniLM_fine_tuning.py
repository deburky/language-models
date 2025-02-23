import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Check if MPS (Apple GPU) is available, otherwise fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load dataset
dataset = pd.read_parquet("data/fine_food_reviews_1k.parquet")

# Prepare labels: Binary classification (1 = Negative, 0 = Positive)
dataset["label"] = (dataset["Score"] < 3).astype(int)
dataset = dataset[["Text", "label"]].rename(columns={"Text": "text"})

# Split into train and validation sets
train_df, val_df = train_test_split(dataset, test_size=0.3, random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
dataset_dict = DatasetDict({"train": train_dataset, "test": val_dataset})

# Load tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Tokenize datasets
tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

# Remove unnecessary columns
tokenized_datasets.set_format("torch")

# Define train and validation datasets
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["test"]

# Load model for binary classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(
    device
)

# Freeze all transformer layers
# (Comment out for fine-tuning of the entire model)
for param in model.base_model.encoder.layer[:4].parameters():
    param.requires_grad = False

# Unfreeze the last few layers for fine-tuning
for param in model.base_model.encoder.layer[4:].parameters():
    param.requires_grad = True

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_steps=10,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
)


# Define compute metrics function
def compute_metrics(p):
    probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=1)[
        :, 1
    ].numpy()
    preds = (probs >= 0.5).astype(int)
    labels = p.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=1
    )
    accuracy = accuracy_score(labels, preds)
    gini = 2 * roc_auc_score(labels, probs) - 1

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gini": gini,
    }


# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Track training time
start_time = time.time()

# Train the model (only the classification head)
trainer.train()

# Training time
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")
