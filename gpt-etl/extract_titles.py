"""extract_titles.py."""

import argparse
import json
import os


def extract_conversation_titles(data_folder):
    """Extract all conversation titles from conversations.json and save to a file.

    Returns:
        tuple: (titles_list, output_file_path)
    """
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversations_file = os.path.join(data_folder, "conversations.json")
    titles_file = os.path.join(script_dir, "conversation_titles.txt")
    titles_json_file = os.path.join(script_dir, "conversation_titles.json")

    # Read the conversations.json file
    with open(conversations_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    print(f"Found {len(conversations)} total conversations")

    # Extract titles with their IDs
    titles_with_ids = []
    for i, conversation in enumerate(conversations):
        title = conversation.get("title", f"Untitled Conversation {i}")
        conversation_id = conversation.get("id", "")
        titles_with_ids.append({"id": conversation_id, "title": title})

    # Sort by title for easier browsing
    titles_with_ids.sort(key=lambda x: x["title"].lower())

    # Save as plain text (for easy reading)
    with open(titles_file, "w", encoding="utf-8") as f:
        for item in titles_with_ids:
            f.write(f"{item['title']}\n")

    # Save as JSON (for programmatic use)
    with open(titles_json_file, "w", encoding="utf-8") as f:
        json.dump(titles_with_ids, f, indent=2)

    print(f"Saved {len(titles_with_ids)} conversation titles to:")
    print(f"- {titles_file} (plain text)")
    print(f"- {titles_json_file} (JSON with IDs)")

    return titles_with_ids, titles_json_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract conversation titles from a specified data folder."
    )
    parser.add_argument(
        "data_folder", help="Path to the folder containing the conversations.json file"
    )
    args = parser.parse_args()

    try:
        titles, json_file = extract_conversation_titles(args.data_folder)
        print("\nExample usage:")
        print("1. Review the titles in conversation_titles.txt")
        print("2. Use extract_conversations.py with specific titles:")
        print('   python extract_conversations.py "Title of Interest"')
        print("\nFirst 5 conversation titles:")
        for i, item in enumerate(titles[:5]):
            print(f"{i + 1}. {item['title']}")
    except Exception as e:
        print(f"Error: {e}")
