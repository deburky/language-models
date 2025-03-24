# Ollama + Pydantic AI

<img src="https://img.shields.io/badge/python-3.9.2-blue.svg" alt="python 3.9.2">

This is a local project that uses Pydantic AI agents with the Ollama library (local LLM). 

The workflow runs locally with Ollama and Pydantic AI agents.

Author: [Denis Burakov @deburky](https://github.com/deburky)

## Installation

Install with `uv`:
```bash
uv sync
```
Run `main.py`:

```bash
uv run main.py
```

## Usage

Send a request to the pipeline endpoint:

```plain
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{"description": "Build a pipeline that loads data from S3, cleans it, applies PCA, and saves it back."}'
```