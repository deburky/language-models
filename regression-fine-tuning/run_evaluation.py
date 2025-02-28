"""run_evaluation.py"""

import os
import time

import numpy as np
import pandas as pd
import torch
import typer
import xgboost as xgb
from loguru import logger
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = typer.Typer()
console = Console()

# Ensure necessary directories exist
os.makedirs("logs", exist_ok=True)

# Setup logging
logger.remove()
logger.add(
    "logs/supervised_evaluation.log",
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


def load_data(data_path: str):
    """Load and preprocess the dataset."""
    logger.info("📥 Loading data...")
    dataframe = pd.read_parquet(data_path).sample(n=100)

    dataframe = dataframe[~dataframe["item_name"].isna()].copy()
    dataframe["asin_value_usd"] = dataframe["asin_value_usd"].astype(float)

    # Normalize price data
    price_pipeline = Pipeline(
        [
            ("outlier_cap", RobustScaler(with_centering=False)),
            ("scaler", MinMaxScaler(feature_range=(-1.0, 1.0))),
        ]
    )
    dataframe["asin_value_usd_norm"] = price_pipeline.fit_transform(
        dataframe["asin_value_usd"].values.reshape(-1, 1)
    )

    logger.success("✅ Data loaded successfully!")
    return dataframe, price_pipeline


def get_embeddings(text_list, model_path, tokenizer, pooling="cls", device="mps"):
    """Load a fine-tuned model and extract embeddings."""
    model_ft = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model_ft.eval()
    embeddings = []

    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_ft(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        if pooling == "cls":
            embedding = last_hidden_states[:, 0, :]
        elif pooling == "mean":
            embedding = last_hidden_states.mean(dim=1)
        else:
            raise ValueError("Pooling must be 'cls' or 'mean'")

        embeddings.append(embedding.cpu().numpy())

    return np.vstack(embeddings)


def set_random_seed(seed: int, device: str):
    """Set seed for NumPy, PyTorch, and device-specific configurations."""
    np.random.seed(seed)
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


@app.command()
def evaluate(
    data_path: str = typer.Option("marketing_sample.parquet", help="Path to dataset"),
    device: str = typer.Option(
        "mps", help="Device for model evaluation (cpu, cuda, mps)"
    ),
    pooling: str = typer.Option("cls", help="Pooling strategy: cls or mean"),
    n_estimators: int = typer.Option(
        100, help="Number of trees in XGBoost Random Forest"
    ),
    seed: int = typer.Option(0, help="Random seed for reproducibility"),
):
    """Run supervised evaluation on fine-tuned models using XGBoost."""
    logger.info(f"🔁 Starting evaluation process with seed {seed}...")
    set_random_seed(seed, device)
    start_time = time.perf_counter()

    dataframe, price_pipeline = load_data(data_path)

    # Load Sentence Transformer & Compute Baseline Embeddings
    logger.info("Generating original embeddings...")
    local_model_path = "models/base_model"
    model = SentenceTransformer(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    dataframe["embedding"] = dataframe["item_name"].apply(
        lambda text: model.encode(text)
    )

    # Check available layers dynamically
    base_model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    num_layers = len(base_model.bert.encoder.layer)
    logger.info(f"Base model has {num_layers} layers.")

    # Define fine-tuned model paths dynamically
    fine_tuning_scenarios = {
        "original_embedding": None,  # No fine-tuning, use original embeddings
        "full_finetuning": "models/full_finetuning",
        "classification_head_only": "models/classification_head_only",
    }

    # Dynamically find available fine-tuned layer models
    existing_model_dirs = os.listdir("models")
    layer_finetuned_models = {
        f"train_from_layer_{n}_finetuning": f"models/train_from_layer_{n}"
        for n in range(1, num_layers + 1)
        if f"train_from_layer_{n}"
        in existing_model_dirs  # Only include existing models
    }

    # Merge dictionaries
    fine_tuning_scenarios |= layer_finetuned_models

    # Warn if some expected models are missing
    expected_layers = [
        f"train_from_layer_{n}_finetuning" for n in range(1, num_layers + 1)
    ]
    if missing_models := [m for m in expected_layers if m not in fine_tuning_scenarios]:
        logger.warning(
            f"⚠️ Some expected fine-tuned models are missing: {missing_models}"
        )

    # Load Fine-Tuned Models & Extract Embeddings
    available_models = {
        k: v for k, v in fine_tuning_scenarios.items() if v is None or os.path.exists(v)
    }

    for scenario_name, model_path in available_models.items():
        step_start = time.perf_counter()

        if scenario_name == "original_embedding":
            logger.info("📥 Using original embeddings for {}", scenario_name)
            dataframe[f"embedding_{scenario_name}"] = list(dataframe["embedding"])
            continue

        logger.info("📥 Extracting embeddings for {}", scenario_name)
        embeddings = get_embeddings(
            dataframe["item_name"].tolist(),
            model_path,
            tokenizer,
            pooling=pooling,
            device=device,
        )
        dataframe[f"embedding_{scenario_name}"] = list(embeddings)

        step_time = time.perf_counter() - step_start
        logger.success(
            "Finished processing {} in {:.2f} seconds", scenario_name, step_time
        )

    # Train & Evaluate XGBoost for Regression
    logger.info("🧩 Training & evaluating supervised regression models...")
    results = []

    for scenario_name in available_models:
        if f"embedding_{scenario_name}" not in dataframe.columns:
            continue  # Skip missing models

        X = np.vstack(dataframe[f"embedding_{scenario_name}"].values)
        y = dataframe["asin_value_usd_norm"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # Train XGBoost Random Forest Regressor
        rf_model = xgb.XGBRFRegressor(n_estimators=n_estimators, random_state=seed)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test).reshape(-1, 1)

        # Inverse transform results
        y_test_inverse = price_pipeline.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inverse = price_pipeline.inverse_transform(y_pred)

        rmse = root_mean_squared_error(y_test_inverse, y_pred_inverse)
        mae = mean_absolute_error(y_test_inverse, y_pred_inverse)

        results.append(
            {"Scenario": scenario_name, "RMSE": round(rmse, 2), "MAE": round(mae, 2)}
        )
        logger.success(
            "Scenario: {}, RMSE: {:.2f}, MAE: {:.2f}", scenario_name, rmse, mae
        )

    total_time = time.perf_counter() - start_time
    logger.success("🧪 Evaluation completed in {:.2f} seconds!", total_time)

    # Display Results in a Rich Table
    table = Table(title="🌲 XGBoost Random Forest Regression Results")
    table.add_column("Scenario", style="cyan1", justify="left")
    table.add_column("RMSE", style="cyan1", justify="right")
    table.add_column("MAE", style="cyan1", justify="right")

    for result in results:
        table.add_row(result["Scenario"], str(result["RMSE"]), str(result["MAE"]))

    console.print("\n[bold cyan1]Final Supervised Learning Results:[/bold cyan1]")
    console.print(table)


if __name__ == "__main__":
    app()
