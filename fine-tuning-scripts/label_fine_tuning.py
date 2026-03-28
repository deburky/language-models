"""Fine-tune E5-small for multi-class star ratings on food reviews."""

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

model_name = "intfloat/e5-small-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_parquet("data/fine_food_reviews_1k.parquet")
df["label"] = (df["Score"].astype(int) - 1).clip(0, 4)
df = df[["Text", "label"]].rename(columns={"Text": "text"})
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))


def tokenize_function(examples):
    """Tokenize review text for the sequence classifier."""
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=128
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=1,
    weight_decay=0.01,
    warmup_steps=10,
    logging_dir="./logs",
    logging_steps=10,
)


def compute_metrics(p):
    """Return accuracy and weighted precision, recall, and F1."""
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="weighted"
    )
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("saved_model/fine_tuned_with_labels")
