"""run_fine_tuning.py"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import typer
from datasets import Dataset
from loguru import logger
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

app = typer.Typer()
console = Console()

# Ensure necessary directories exist
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Setup logging
logger.remove()
logger.add(
    "logs/fine_tuning.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="1 MB",
    retention="10 days",
    compression="zip",
)
logger.add(
    sink=lambda msg: print(msg, end=""),  # Console output
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True,
)


def load_data(data_path: str) -> pd.DataFrame:
    """Load the dataset and preprocess it."""
    logger.info("📂 Loading data...")
    df = pd.read_parquet(data_path).sample(100)
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

    logger.success("✅ Data loaded and preprocessed successfully!")
    return df, price_pipeline


def load_or_download_model():
    """Load MiniLM model from disk or download it."""
    base_model_dir = "models/base_model"
    if not os.path.exists(base_model_dir):
        logger.info("📥 Downloading & saving base MiniLM model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        model.save(base_model_dir)
        tokenizer.save_pretrained(base_model_dir)
        logger.success(f"✅ Base MiniLM model saved to `{base_model_dir}`")
    else:
        logger.success(f"✅ Loading Base Model from `{base_model_dir}`...")
        model = SentenceTransformer(base_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    return model, tokenizer


def prepare_dataset(dataframe: pd.DataFrame, tokenizer) -> tuple:
    """Prepare dataset for fine-tuning."""
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
    """Freeze all layers except the classification head."""
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model


def freeze_above_n_layers(model, n: int):
    """Freeze all layers below `n`, keep `n` and above trainable."""
    for param in model.bert.encoder.layer[:n].parameters():
        param.requires_grad = False
    for param in model.bert.encoder.layer[n:].parameters():
        param.requires_grad = True
    return model


def count_trainable_parameters(model):
    return round(sum(p.numel() for p in model.parameters() if p.requires_grad), 0)


def set_random_seed(seed: int, device: str):
    """Set seed for NumPy, PyTorch, and device-specific configurations."""
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device == "mps":
        torch.mps.manual_seed(seed)
        torch.mps.set_per_process_memory_fraction(0.7)
        torch.backends.mps.allow_tf32 = True
    else:
        torch.manual_seed(seed)


def compute_metrics(eval_pred):
    """Compute RMSE and MAE."""
    predictions, labels = eval_pred
    predictions = np.array(predictions).reshape(-1, 1)
    labels = np.array(labels).reshape(-1, 1)

    rmse = root_mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)

    return {"rmse": rmse, "mae": mae}


@app.command()
def fine_tune(
    data_path: str = typer.Option("marketing_sample.parquet", help="Path to dataset"),
    batch_size: int = typer.Option(16, help="Batch size for training"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    learning_rate: float = typer.Option(5e-5, help="Learning rate"),
    device: str = typer.Option("mps", help="Training device (cpu, cuda, mps)"),
    freeze_layers: int = typer.Option(
        None,
        help="Specify which layer to start training from. Freezes all below this layer.",
    ),
    clf_only: bool = typer.Option(
        True, help="If set, trains only the classification head."
    ),
    full: bool = typer.Option(True, help="If set, includes full fine-tuning as well."),
    seed: int = typer.Option(0, help="Random seed for reproducibility"),
):
    """Run fine-tuning experiments with flexible layer freezing."""
    logger.info(f"🌎 Starting fine-tuning process with seed {seed}...")
    set_random_seed(seed, device)
    start_time = time.perf_counter()

    dataframe, price_pipeline = load_data(data_path)
    model, tokenizer = load_or_download_model()

    logger.info("📥 Generating embeddings...")
    dataframe["embedding"] = dataframe["item_name"].apply(
        lambda text: model.encode(text)
    )

    train_dataset, eval_dataset = prepare_dataset(dataframe, tokenizer)

    # Load base model to check the number of layers
    temp_model = AutoModelForSequenceClassification.from_pretrained("models/base_model")
    num_layers = len(temp_model.bert.encoder.layer)
    logger.info(f"Model has {num_layers} transformer layers.")
    logger.info(
        f"Trainable parameters for base model: {count_trainable_parameters(temp_model):,.0f}"
    )

    for layer in range(num_layers + 1):
        logger.info(f"Checking: train_from_layer_{layer} will be included.")

    results = []
    fine_tuning_scenarios = {}

    # Add full fine-tuning if enabled
    if full:
        fine_tuning_scenarios["full_finetuning"] = lambda model: model

    # Add classification head only if enabled
    if clf_only:
        fine_tuning_scenarios["classification_head_only"] = (
            lambda model: freeze_all_but_classifier(model)
        )

    # If a layer freeze is specified, add scenarios for all layers from that point onward
    if freeze_layers is not None:
        if freeze_layers >= num_layers:
            logger.warning(
                f"Requested freeze_layers={freeze_layers}, but model only has {num_layers} layers. Skipping..."
            )
            return

        for layer in range(freeze_layers, num_layers + 1):  # Includes the last layer
            fine_tuning_scenarios[f"train_from_layer_{layer}"] = (
                lambda model, layer=layer: freeze_above_n_layers(model, layer)
            )

    for scenario_name, scenario_function in fine_tuning_scenarios.items():
        logger.info(f"🪛 Running fine-tuning with {scenario_name}")

        model_ft = AutoModelForSequenceClassification.from_pretrained(
            "models/base_model", num_labels=1
        ).to(device)
        model_ft = scenario_function(model_ft)
        logger.info(
            f"Trainable parameters for {scenario_name}: {count_trainable_parameters(model_ft):,.0f}"
        )

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

        scenario_start = time.perf_counter()
        trainer.train()
        training_time = time.perf_counter() - scenario_start

        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        eval_results = trainer.evaluate()
        logger.success(
            f"Finished training {scenario_name} in {training_time:.2f} seconds"
        )

        results.append(
            {
                "Scenario": scenario_name,
                "RMSE": round(eval_results["eval_rmse"], 2),
                "MAE": round(eval_results["eval_mae"], 2),
                "Time (min)": round(training_time / 60, 2),
            }
        )

    logger.success(
        f"🧪 Fine-tuning completed in {time.perf_counter() - start_time:.2f} seconds!"
    )

    # Save results to json
    with open("logs/fine_tuning_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    app()
