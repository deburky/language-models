# 📖 Regression fine-tuning

This project automates the fine-tuning of sentence embeddings for regression tasks and evaluates them using supervised learning (XGBoost Random Forest). 

The idea behind this is to make embeddings context-aware in terms of price or another numerical attribute.

We use an [open-source dataset](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020) with Amazon product descriptions and price.

Since our target is numerical, we normalize the target variable (asin_value_usd) to improve model performance and stability.

* Cap Outliers → `RobustScaler` to reduce the impact of extreme values.
* Normalize the Range → `MinMaxScaler` to scale prices between (-1.0, 1.0).

## Overview

1️⃣ `run_fine_tuning_experiments.py`

- Fine-tunes a MiniLM sentence transformer on pricing data.
- Supports multiple fine-tuning strategies (e.g., last N layers, full model, classification head).
- Saves fine-tuned models for later use.

2️⃣ `run_supervised_evaluation.py`

- Loads the fine-tuned models.
- Extracts embeddings from the trained models.
- Evaluates performance using XGBoost RF regression on extracted embeddings.

3️⃣ `run_full_pipeline.py`

- Master script that runs both fine-tuning and evaluation sequentially.
- Ensures that fine-tuning is completed before evaluation starts.
- Displays live output logs using `rich` library.

## Results

After training and evaluation, the pipeline displays:

✅ Validation RMSE & MAE (during training)  
✅ Supervised Regression RMSE & MAE (on extracted embeddings)  

📊 Final results are presented in a Rich table.

```plain
📂 regression-fine-tuning/
│── 📜 run_fine_tuning_experiments.py
│── 📜 run_supervised_evaluation.py
│── 📜 run_full_pipeline.py
│── 📜 requirements.txt
│── 📜 README.md
│── 📂 models/
```