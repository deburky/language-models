#!/usr/bin/env python3
import argparse
import json
import os
import subprocess


def run_script(script_name, *args):
    """Run a Python script with the given arguments."""
    command = ["uv", "run", script_name] + list(args)
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def main(data_folder):
    """Main function to run the ETL process."""
    # Run extract_titles.py
    print("Running extract_titles.py...")
    run_script("extract_titles.py", data_folder)

    # Load titles from conversation_titles.json
    titles_json_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "conversation_titles.json"
    )
    with open(titles_json_file, "r", encoding="utf-8") as f:
        titles = json.load(f)

    # Run extract_conversations.py for each title
    for title_obj in titles:
        print(f"Extracting conversation: {title_obj['title']}")
        run_script("extract_conversations.py", data_folder, "--id", title_obj["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ETL process for extracting and converting conversations."
    )
    parser.add_argument(
        "data_folder", help="Path to the folder containing the conversations.json file"
    )
    args = parser.parse_args()

    main(args.data_folder)
