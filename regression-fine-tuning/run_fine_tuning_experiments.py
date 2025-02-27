"""run_fine_tuning_experiments.py."""

import logging
import os
import time
import json

import numpy as np
import pandas as pd
import torch
import typer
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

app = typer.Typer()
console = Console()

# Ensure directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Setup logging
logging.basicConfig(
    filename="logs/fine_tuning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def load_data(data_path: str) -> pd.DataFrame:
    """Loads the dataset and preprocesses it."""
    console.print("\n📥 [bold cyan1]Loading Data...[/bold cyan1]")
    df = pd.read_parquet(data_path)
    df = df[~df["item_name"].isna()].copy()
    df["asin_value_usd"] = df["asin_value_usd"].astype(float)

    # Normalize price data
    price_pipeline = Pipeline(
        [
            ("outlier_cap", RobustScaler(with_centering=False)),
            ("scaler", MinMaxScaler(feature_range=(-1.0, 1.0))),
        ]
    )
    df["asin_value_usd_norm"] = price_pipeline.fit_transform(
        df["asin_value_usd"].values.reshape(-1, 1)
    )

    return df, price_pipeline


def load_or_download_model():
    """Loads MiniLM model from disk or downloads it."""
    base_model_dir = "models/base_model"
    if not os.path.exists(base_model_dir):
        console.print(
            "\n🔍 [bold cyan1]Downloading & Saving Base MiniLM Model...[/bold cyan1]"
        )
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        model.save(base_model_dir)
        tokenizer.save_pretrained(base_model_dir)
        console.print(f"✅ Base MiniLM model saved to `{base_model_dir}`")
    else:
        console.print(
            f"\n✅ [bold green]Loading Base Model from `{base_model_dir}`...[/bold green]"
        )
        model = SentenceTransformer(base_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)

    return model, tokenizer


def prepare_dataset(dataframe: pd.DataFrame, tokenizer) -> tuple:
    """Prepares dataset for fine-tuning."""
    df = dataframe[["item_name", "asin_value_usd_norm"]]
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(
            examples["item_name"], padding="max_length", truncation=True, max_length=128
        )

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.map(
        lambda e: {"labels": torch.tensor(e["asin_value_usd_norm"], dtype=torch.float)}
    )
    dataset = dataset.remove_columns(["item_name", "asin_value_usd_norm"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    split = dataset.train_test_split(test_size=0.2)
    return split["train"], split["test"]


def freeze_all_but_classifier(model):
    """Freezes all layers except the classification head."""
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def freeze_below_n_layers(model, n: int):
    """Freezes all layers below `n`."""
    for param in model.bert.encoder.layer[:n].parameters():
        param.requires_grad = False
    for param in model.bert.encoder.layer[n:].parameters():
        param.requires_grad = True
    return model


@app.command()
def fine_tune(
    data_path: str = typer.Option(
        "marketing_sample_for_amazon_com.parquet", help="Path to data"
    ),
    batch_size: int = typer.Option(16, help="Batch size for training"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate"),
    device: str = typer.Option("mps", help="Training device (cpu, cuda, mps)"),
):
    """Runs fine-tuning experiments with different layer freezing strategies."""
    dataframe, price_pipeline = load_data(data_path)
    model, tokenizer = load_or_download_model()

    # Generate embeddings
    console.print("\n[bold cyan1]Generating Embeddings...[/bold cyan1]")
    dataframe["embedding"] = dataframe["item_name"].apply(
        lambda text: model.encode(text)
    )

    train_dataset, eval_dataset = prepare_dataset(dataframe, tokenizer)

    # Fine-Tuning strategies
    fine_tuning_scenarios = {
        "full_finetuning": lambda model: model,
        "classification_head_only": lambda model: freeze_all_but_classifier(model),
        **{
            f"train_from_layer_{n}": (
                lambda model, n=n: freeze_below_n_layers(model, n)
            )
            for n in range(1, 6)
        },
    }

    # Metric computation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.array(predictions).reshape(-1, 1)
        labels = np.array(labels).reshape(-1, 1)
        rmse = root_mean_squared_error(
            price_pipeline.inverse_transform(labels),
            price_pipeline.inverse_transform(predictions),
        )
        mae = mean_absolute_error(
            price_pipeline.inverse_transform(labels),
            price_pipeline.inverse_transform(predictions),
        )
        return {"rmse": rmse, "mae": mae}

    # Run fine-tuning
    results = []
    for scenario_name, scenario_function in fine_tuning_scenarios.items():
        console.print(
            f"\n🚀 [bold cyan1]Running Scenario:[/bold cyan1] {scenario_name}"
        )
        logger.info(f"Starting fine-tuning: {scenario_name}")

        model_ft = AutoModelForSequenceClassification.from_pretrained(
            "models/base_model", num_labels=1
        ).to(device)
        model_ft = scenario_function(model_ft)

        output_dir = f"models/{scenario_name}"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir="logs",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model_ft,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        # Save model and tokenizer
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        eval_results = trainer.evaluate()
        training_time = end_time - start_time
        results.append(
            [
                scenario_name,
                round(eval_results["eval_rmse"], 2),
                round(eval_results["eval_mae"], 2),
                round(training_time / 60, 2),
            ]
        )

    # Print results
    table = Table(title="Fine-Tuning Results")
    table.add_column("Scenario", style="cyan1", justify="left")
    table.add_column("RMSE", style="cyan1", justify="right")
    table.add_column("MAE", style="cyan1", justify="right")
    table.add_column("Time (min)", style="cyan1", justify="right")

    for row in results:
        table.add_row(*map(str, row))

    console.print("\n[bold cyan1]Final Fine-Tuning Results:[/bold cyan1]")
    console.print(table)

    # Log the table output
    logger.info("Fine-Tuning Results:\n" + json.dumps(results, indent=4))

if __name__ == "__main__":
    app()
