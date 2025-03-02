"""create_db.py."""

import duckdb
import os
import numpy as np
from transformers import pipeline
from pydantic import BaseModel, field_validator, ValidationError

# Initialize DuckDB connection
con = duckdb.connect("db/embeddings.db")

# Initialize the feature-extraction pipeline with the pre-trained model
model_name = "distilbert-base-uncased-distilled-squad"

embedding_pipeline = pipeline(
    "feature-extraction",  # Feature extraction task (embedding extraction)
    model=model_name,
    tokenizer=model_name,
)

# Create table for embeddings if it doesn't exist
con.execute(
    """
    DROP TABLE IF EXISTS embeddings;
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,
        text VARCHAR,
        embedding FLOAT[768]
    )
"""
)

# Function to generate embeddings using the pipeline
def generate_embeddings(text):
    # Use the pipeline to extract the embeddings
    embeddings = embedding_pipeline(text)
    # Take the mean of the embeddings
    return np.mean(embeddings[0], axis=0)


# Pydantic model to validate each line
class TextLine(BaseModel):
    text: str

    @field_validator("text")
    def validate_text(cls, value):
        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("Line cannot be empty.")
        if len(stripped_value.split()) < 2:
            raise ValueError("Line must contain at least two words.")
        return stripped_value


# Generator function to track and yield a sequential index
def index_generator(start=0):
    index = start
    while True:
        yield index
        index += 1


# Read text from the file
file_path = "documents/fastapi_docs.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

with open(file_path, "r") as file:
    lines = file.readlines()

# Create the index generator
index_gen = index_generator()

# Insert embeddings into DuckDB
for line in lines:
    try:
        # Validate and generate embeddings for each line
        valid_line = TextLine(text=line)
        embedding = generate_embeddings(valid_line.text)
        valid_index = next(index_gen)

        # Insert text and its embedding into DuckDB
        con.execute(
            "INSERT INTO embeddings (id, text, embedding) VALUES (?, ?, ?)",
            (valid_index, valid_line.text, embedding.tolist()),
        )
    except ValidationError as e:
        # If the line is invalid (empty or doesn't have at least two words), skip it
        print(f"Skipping invalid line. Reason: {e.errors()[0]['msg']}")

con.close()

print("Embeddings have been generated and stored in DuckDB.")
