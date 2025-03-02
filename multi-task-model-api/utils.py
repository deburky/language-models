"""utils.py."""

import os
import typer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

app = typer.Typer()


@app.command()
def download_model(
    model_name=typer.Option(
        "distilbert/distilgpt2", help="The name of the model to download"
    ),
):
    model_path = f"./models/{model_name}"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        print("Downloading model...")
        # Download the model and tokenizer from Hugging Face
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model downloaded and saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")


if __name__ == "__main__":
    app()
