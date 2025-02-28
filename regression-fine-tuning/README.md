# 🧪 Regression fine-tuning

This project automates the fine-tuning of sentence embeddings for regression tasks and evaluates them using supervised learning (XGBoost Random Forest). 

The idea behind this is to make embeddings context-aware in terms of price or another numerical attribute.

We use an [open-source dataset](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020) with Amazon product descriptions and price.

Since our target is numerical, we normalize the target variable (asin_value_usd) to improve model performance and stability.

* Cap Outliers → `RobustScaler` to reduce the impact of extreme values.
* Normalize the Range → `MinMaxScaler` to scale numerical target between (-1.0, 1.0).

## Overview

1️⃣ **`run_fine_tuning_experiments.py`**
- Supports multiple fine-tuning strategies, including:
    - Full fine-tuning (entire model)
    - Training from a specific layer (--freeze-layers=N)
    - Only fine-tuning the classification head.

2️⃣ **`run_supervised_evaluation.py`**
- Extracts embeddings from the trained transformers.
- Evaluates performance using XGBoost RF regression on extracted embeddings.

3️⃣ **`run_full_pipeline.py`**
- Master script that runs both fine-tuning and evaluation sequentially.
- Ensures that fine-tuning is completed before evaluation starts.

## Results

Run as follows:

```bash
uv run run_pipeline.py
```

After training and evaluation, the pipeline displays:

✅ Validation RMSE & MAE (during training)  
✅ Supervised Regression RMSE & MAE (on extracted embeddings)  

```plain
📂 regression-fine-tuning/
│── 📄 run_fine_tuning.py       # Fine-tune embeddings
│── 📄 run_evaluation.py        # Evaluate trained embeddings
│── 📄 run_pipeline.py          # Automates full pipeline
│── 📄 requirements.txt         # Python dependencies
│── 📄 README.md                # Project documentation
│── 📂 models/                  # Trained models
│── 📂 logs/                    # Experiment logs
│── 📄 pyproject.toml           # uv dependency management
```