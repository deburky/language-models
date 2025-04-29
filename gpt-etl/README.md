# GPT-ETL

This project provides tooling for storing and searching embeddings using Chroma DB. It includes scripts for embedding, storing, and searching data, as well as an orchestration flow using Prefect.

## Installation

To install the project, ensure you have `uv` installed. You can install it using:

```bash
pip install uv
```

## Scripts

### extract_titles.py

This script extracts titles from the data and prepares them for embedding.

#### Usage

To run the script and extract titles, use the following command:

```bash
uv run extract_titles.py
```

### extract_conversations.py

This script extracts conversations from the data and prepares them for embedding.

#### Usage

To run the script and extract conversations, use the following command:

```bash
uv run extract_conversations.py
```

### etl_to_db.py

This script is responsible for embedding markdown files and storing them in the Chroma DB.

#### Usage

To run the script and store embeddings, use the following command:

```bash
uv run etl_to_db.py
```

### search_db.py

This script allows you to search the Chroma DB for the most similar documents to a given query.

#### Usage

To perform a search, use the following command:

```bash
uv run search_db.py
```

You can modify the query in the script to search for different terms.

### orchestrate_flow.py

This script uses Prefect to orchestrate the entire workflow, including extracting titles, extracting conversations, running the ETL process, and searching the database.

#### Usage

To run the entire workflow, use the following command:

```bash
uv run orchestrate_flow.py
```

## Requirements

Ensure you have the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```
or

```bash
uv sync
```

## Notes

- The embeddings are stored in a local directory named `db`.
- Ensure the `conversations` directory contains the markdown files you wish to embed.
