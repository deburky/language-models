"""run_supervised_evaluation.py"""

import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import typer
import xgboost as xgb
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
logging.basicConfig(
    filename="logs/supervised_evaluation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def load_data(data_path: str):
    """Loads and preprocesses the dataset."""
    console.print("\n📥 [bold cyan1]Loading Data...[/bold cyan1]")
    dataframe = pd.read_parquet(data_path)

    dataframe = dataframe[~dataframe["item_name"].isna()].copy()
    dataframe["asin_value_usd"] = dataframe["asin_value_usd"].astype(float)

    # Normalize Price Data
    price_pipeline = Pipeline(
        [
            ("outlier_cap", RobustScaler(with_centering=False)),
            ("scaler", MinMaxScaler(feature_range=(-1.0, 1.0))),
        ]
    )
    dataframe["asin_value_usd_norm"] = price_pipeline.fit_transform(
        dataframe["asin_value_usd"].values.reshape(-1, 1)
    )

    return dataframe, price_pipeline


def get_embeddings(text_list, model_path, tokenizer, pooling="cls", device="mps"):
    """Loads a fine-tuned model and extracts embeddings."""
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


@app.command()
def evaluate(
    data_path: str = typer.Option(
        "marketing_sample_for_amazon_com.parquet", help="Dataset"
    ),
    device: str = typer.Option(
        "mps", help="Device for model evaluation (cpu, cuda, mps)"
    ),
    pooling: str = typer.Option("cls", help="Pooling strategy: cls or mean"),
    n_estimators: int = typer.Option(
        100, help="Number of trees in XGBoost Random Forest"
    ),
):
    """Runs supervised evaluation on fine-tuned models using XGBoost."""
    dataframe, price_pipeline = load_data(data_path)

    # Load Sentence Transformer & Compute Baseline Embeddings
    console.print("\n🔍 [bold cyan1]Generating Original Embeddings...[/bold cyan1]")
    # Load model from local path
    local_model_path = "models/base_model"
    model = SentenceTransformer(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    dataframe["embedding"] = dataframe["item_name"].apply(
        lambda text: model.encode(text)
    )

    # Define fine-tuned model paths
    fine_tuning_scenarios = {
        "original_embedding": None,  # No fine-tuning, use original embeddings
        "full_finetuning": "models/full_finetuning",
        "classification_head_only": "models/classification_head_only",
        **{
            f"last_{n}_layers_finetuning": f"models/train_from_layer_{n}"
            for n in range(1, 6)
        },
    }

    # Load Fine-Tuned Models & Extract Embeddings
    console.print(
        "\n🔍 [bold cyan1]Loading Fine-Tuned Models & Extracting Embeddings...[/bold cyan1]"
    )

    for scenario_name, model_path in fine_tuning_scenarios.items():
        if scenario_name == "original_embedding":
            console.print(f"\n🔍 Using original embeddings for {scenario_name}")
            dataframe[f"embedding_{scenario_name}"] = list(dataframe["embedding"])
            continue  # Skip model loading for original embeddings

        if os.path.exists(model_path):  # Ensure model exists
            console.print(
                f"\n🔍 Extracting embeddings for {scenario_name} with `{pooling}` pooling"
            )
            embeddings = get_embeddings(
                dataframe["item_name"].tolist(),
                model_path,
                tokenizer,
                pooling=pooling,
                device=device,
            )
            dataframe[f"embedding_{scenario_name}"] = list(embeddings)
        else:
            console.print(
                f"\n[red]Model for {scenario_name} not found! Skipping...[/red]"
            )

    # Train & Evaluate XGBoost for Regression
    console.print(
        "\n📊 [bold cyan1]Training & Evaluating Supervised Regression Models...[/bold cyan1]"
    )
    results = []

    for scenario_name in fine_tuning_scenarios:
        if f"embedding_{scenario_name}" not in dataframe.columns:
            continue  # Skip missing models

        X = np.vstack(dataframe[f"embedding_{scenario_name}"].values)
        y = dataframe["asin_value_usd_norm"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train XGBoost Random Forest Regressor
        rf_model = xgb.XGBRFRegressor(n_estimators=n_estimators, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test).reshape(-1, 1)

        # Inverse transform results
        y_test_inverse = price_pipeline.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inverse = price_pipeline.inverse_transform(y_pred)

        rmse = root_mean_squared_error(y_test_inverse, y_pred_inverse)
        mae = mean_absolute_error(y_test_inverse, y_pred_inverse)

        results.append(
            {"Scenario": scenario_name, "RMSE": round(rmse, 4), "MAE": round(mae, 4)}
        )
        logger.info(f"Scenario: {scenario_name}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Display Results in a Rich Table
    table = Table(title="XGBoost Random Forest Regression Results")
    table.add_column("Scenario", style="cyan1", justify="left")
    table.add_column("RMSE", style="cyan1", justify="right")
    table.add_column("MAE", style="cyan1", justify="right")

    for result in results:
        table.add_row(result["Scenario"], str(result["RMSE"]), str(result["MAE"]))

    console.print("\n📊 [bold cyan1]Final Supervised Learning Results:[/bold cyan1]")
    console.print(table)


if __name__ == "__main__":
    app()
