# main.py

import time
import requests
import logging
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import make_column_selector
from catboost import CatBoostClassifier
from scipy.special import logit
from openai import OpenAI
from rich.console import Console
from rich.live import Live
import pandas as pd

from src.utils import (
    pyarrow_to_csv_buffer,
    fetch_messages,
    AssistantRunner,
    delete_all_assistants_and_files,
)

# --- Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
console = Console(highlight=False)
client = OpenAI()
runner = AssistantRunner(client)

# --- Constants ---
DATA_URL = "https://drive.google.com/file/d/1r6gGvL_s313ThGSU7ziZiuYr2G_yijaZ/view?usp=sharing"
DATA_DIR = Path("data")
REPORT_DIR = Path("report")
INSTRUCTIONS_DIR = Path("instructions")
PARQUET_TRAIN = DATA_DIR / "loans-9309cbc146a4.parquet"
PARQUET_TEST = DATA_DIR / "loans-new-9309cbc146a4.parquet"
PARQUET_SCORES = DATA_DIR / "scores-new-9309cbc146a4.parquet"

# --- Step Functions ---


def fetch_and_prepare_data():
    logger.info("📥 Downloading and preparing loan data...")
    file_id = DATA_URL.split("/")[-2]
    response = requests.get(f"https://drive.google.com/uc?id={file_id}")
    full_data = pacsv.read_csv(BytesIO(response.content)).to_pandas()

    train = (
        full_data.query("Credit_History.notna()")
        .sample(700, random_state=42)
        .reset_index(drop=True)
    )
    test = (
        full_data.query("Credit_History.isnull()")
        .sample(100, random_state=42)
        .reset_index(drop=True)
    )

    train.drop(columns=["Gender"], inplace=True)
    test_with_labels = test.copy()
    test.drop(columns=["Gender", "Loan_Status"], inplace=True)

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    pq.write_table(pa.Table.from_pandas(train), PARQUET_TRAIN)
    pq.write_table(pa.Table.from_pandas(test), PARQUET_TEST)

    return train, test, test_with_labels


def train_model(train_df):
    logger.info("🤖 Training credit model...")
    X = train_df.drop(columns=["Loan_Status", "Loan_ID"])
    y = train_df["Loan_Status"]
    cat_features = make_column_selector(dtype_include="object")(X)

    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=1,
        l2_leaf_reg=3,
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False,
        random_seed=0,
        cat_features=cat_features,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    model.fit(X_train, y_train)

    def gini(score):
        """Calculate Gini from ROC AUC score."""
        return 2 * score - 1

    train_gini = gini(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    val_gini = gini(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

    logger.info(f"✅ Train Gini: {train_gini:.4f}, Validation Gini: {val_gini:.4f}")
    return model, list(X.columns)


def score_test_set(model, features, test_df):
    logger.info("📊 Scoring test dataset...")
    test_df["Score"] = model.predict_proba(test_df[features])[:, 0]
    test_df["Credit_Score"] = -test_df["Score"].apply(lambda x: 350 * logit(x - 0.5))
    test_df.drop(columns=["Score"], inplace=True)
    pq.write_table(pa.Table.from_pandas(test_df), PARQUET_SCORES)

    return test_df


def run_assistants(train_parquet, test_parquet, scores_parquet):
    logger.info("🤝 Uploading files and running assistants...")

    train_file = client.files.create(
        file=("train.csv", pyarrow_to_csv_buffer(pq.read_table(train_parquet))),
        purpose="assistants",
    )
    test_file = client.files.create(
        file=("test.csv", pyarrow_to_csv_buffer(pq.read_table(test_parquet))),
        purpose="assistants",
    )
    scores_file = client.files.create(
        file=("test_with_scores.csv", pyarrow_to_csv_buffer(pq.read_table(scores_parquet))),
        purpose="assistants",
    )

    with open(INSTRUCTIONS_DIR / "instructions_underwriter.txt", "r") as f:
        underwriter_instructions = f.read()
    with open(INSTRUCTIONS_DIR / "instructions_decision_maker.txt", "r") as f:
        decision_instructions = f.read()

    underwriter = client.beta.assistants.create(
        name="Underwriter",
        model="gpt-4o",
        instructions=underwriter_instructions,
        tools=[{"type": "code_interpreter"}],
        tool_resources={"code_interpreter": {"file_ids": [train_file.id, test_file.id]}},
    )
    decision_maker = client.beta.assistants.create(
        name="Decision Maker",
        model="gpt-4o",
        instructions=decision_instructions,
        tools=[{"type": "code_interpreter"}],
        tool_resources={"code_interpreter": {"file_ids": [scores_file.id]}},
    )

    thread = client.beta.threads.create()
    steps = [
        ("📝 Step 1: Underwriter Intro", underwriter.id, "Briefly introduce yourself."),
        ("📝 Step 2: Decision Maker Intro", decision_maker.id, "Introduce yourself."),
        ("📝 Step 3: Final Decision", decision_maker.id, "Review report and return a CSV with Loan_ID and Loan_Status"),
    ]

    with Live(console=console, refresh_per_second=1) as live:
        for step_name, assistant_id, instruction in steps:
            console.print(f"\n{step_name}", style="bold cyan")
            run = runner.start(assistant_id, thread.id, instruction)
            while True:
                live.update(fetch_messages(client, thread.id))
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id, run_id=run.id
                )
                if run_status.status in ["completed", "failed", "cancelled"]:
                    break
                time.sleep(2)

    console.print("\n[bold green]✅ Assistant workflow complete.[/bold green]")
    record_console = Console(record=True)
    record_console.print(fetch_messages(client, thread.id))
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
    record_console.save_html(REPORT_DIR / "underwriting_conversation.html")


def evaluate_assistant_output(test_labels_df):
    logger.info("📈 Evaluating assistant predictions...")

    files = client.files.list()
    file_info = [(file.id, file.filename) for file in files.data]
    df_files = pd.DataFrame(file_info, columns=["File ID", "Filename"])
    
    # Show files in df_files
    logger.info(f"📝 Files in df_files:\n{df_files}")

    output_file_id = df_files[df_files["Filename"].str.endswith(".csv")].iloc[0]["File ID"]
    content = client.files.retrieve_content(output_file_id)

    csv_file = BytesIO(content.encode("utf-8"))
    df = pd.read_csv(csv_file)

    df_eval = pd.DataFrame({
        "Loan_ID": test_labels_df["Loan_ID"].reset_index(drop=True),
        "Loan_Status_Assistant": df["Loan_Status"].reset_index(drop=True),
        "Loan_Status_Truth": test_labels_df["Loan_Status"].reset_index(drop=True),
    })
    df_eval.replace({"Y": 1, "N": 0}, inplace=True)

    decision_matrix = pd.crosstab(
        df_eval["Loan_Status_Assistant"],
        df_eval["Loan_Status_Truth"],
        normalize="all"
    )

    console.print("\n[bold]📊 Confusion Matrix (Assistant vs. Truth)[/bold]")
    console.print(decision_matrix)

    console.print("\n[bold]📈 Approval Rates[/bold]")
    console.print("Assistant approval rate:", round(df_eval["Loan_Status_Assistant"].mean(), 2))
    console.print("Historical approval rate:", round(df_eval["Loan_Status_Truth"].mean(), 2))

    match_rate = decision_matrix.get(1, {}).get(1, 0)
    console.print(f"\n[bold]✅ Matched Approvals:[/bold] {round(match_rate * 100)}%")

    return df_eval


# --- Entry Point ---

def main():
    train_df, test_df, test_labels = fetch_and_prepare_data()
    model, features = train_model(train_df)
    score_test_set(model, features, test_df)
    run_assistants(PARQUET_TRAIN, PARQUET_TEST, PARQUET_SCORES)
    evaluate_assistant_output(test_labels)
    logger.info("🧹 Cleaning up assistants and uploaded files...")
    delete_all_assistants_and_files()


if __name__ == "__main__":
    main()
