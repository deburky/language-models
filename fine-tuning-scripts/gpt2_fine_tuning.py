import pprint
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

# --- Model --- #

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Use `AutoModel` to load `GPT2Model` instead of `GPT2LMHeadModel`
gpt2_model = AutoModel.from_pretrained(model_name)


# Properly define GPT-2 classification model
class GPT2ForClassification(nn.Module):
    def __init__(self, gpt2_model, num_labels=2):
        super(GPT2ForClassification, self).__init__()
        self.gpt2 = gpt2_model
        self.config = gpt2_model.config
        self.config.pad_token_id = tokenizer.pad_token_id  # Set PAD token
        # Add classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]

        logits = self.classifier(last_hidden_state)  # Pass through classifier

        loss = None
        if labels is not None:
            labels = labels.view(-1)
            loss = self.loss_fn(logits, labels)

        # Return a `SequenceClassifierOutput` object
        return SequenceClassifierOutput(loss=loss, logits=logits)


# Wrap GPT-2 with classification head
model_gpt2 = GPT2ForClassification(gpt2_model, num_labels=2).to("mps")

for param in model_gpt2.parameters():
    param.requires_grad = False  # Freeze all

# Freeze all layers except last layer + classifier
for param in model_gpt2.gpt2.h[-3:].parameters():
    param.requires_grad = True

for name, param in model_gpt2.named_parameters():
    if "classifier" in name:
        param.requires_grad = True  # Unfreeze last transformer block + classifier
        print(f"{name} is trainable ✅")

# --- Dataset --- #

# Load dataset
dataset_path = "data/fine_food_reviews_1k.parquet"
df = pd.read_parquet(dataset_path)

# Convert labels: 1 = Negative (Score < 3), 0 = Positive
df["label"] = (df["Score"] < 3).astype(int)
df = df[["Text", "label"]].rename(columns={"Text": "text", "label": "labels"})

# Split into train/test sets
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# --- Training --- #


# Tokenize for GPT-2 (Right padding for decoder models)
def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=120
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# Define metric computation
def compute_metrics(pred):
    logits, labels = pred
    probs = F.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

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


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_gpt2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs_gpt2",
    logging_steps=10,
    load_best_model_at_end=True,
)

# Create Trainer
trainer = Trainer(
    model=model_gpt2,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
start_time = time.time()
trainer.train()

# Training time
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Print evaluation metrics
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(trainer.evaluate())
