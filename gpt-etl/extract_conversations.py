"""extract_conversations.py."""

import argparse
import json
import os
import re


def sanitize_filename(title):
    """Convert title to a valid filename by removing invalid characters."""
    # Replace invalid filename characters with underscores
    # Convert to lowercase and replace spaces with underscores
    sanitized_title = title.lower().replace(" ", "_")
    # Replace invalid filename characters with underscores
    return re.sub(r'[\\/*?:"<>|]', "_", sanitized_title)


def extract_conversations(data_folder, search_title=None, conversation_id=None):
    """Extract conversations from conversations.json and save as markdown files.

    Args:
        search_title (str, optional): If provided, only extract conversations with titles
                                     containing this string (case-insensitive).
        conversation_id (str, optional): If provided, only extract the conversation with
                                        this specific ID.
    """
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversations_dir = os.path.join(script_dir, "conversations")
    conversations_file = os.path.join(data_folder, "conversations.json")

    # Create conversations directory if it doesn't exist
    os.makedirs(conversations_dir, exist_ok=True)

    # Read the conversations.json file
    with open(conversations_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    print(f"Found {len(conversations)} total conversations")

    # Filter conversations based on search criteria
    if conversation_id:
        # Filter by exact conversation ID
        filtered_conversations = []
        for conversation in conversations:
            if conversation.get("id") == conversation_id:
                filtered_conversations.append(conversation)
                break

        conversations = filtered_conversations
        print(f"Filtered to conversation with ID: {conversation_id}")

    elif search_title:
        # Filter by title containing the search string
        search_title_lower = search_title.lower()
        filtered_conversations = []
        for conversation in conversations:
            title = conversation.get("title", "")
            if title and search_title_lower in title.lower():
                filtered_conversations.append(conversation)

        conversations = filtered_conversations
        print(
            f"Filtered to {len(conversations)} conversations containing '{search_title}'"
        )

    # Process each conversation
    for i, conversation in enumerate(conversations):
        try:
            # Extract title and create a sanitized filename
            title = conversation.get("title", f"Conversation_{i}")
            sanitized_title = sanitize_filename(title)
            filename = os.path.join(conversations_dir, f"{sanitized_title}.md")

            # Create markdown content
            md_content = f"# {title}\n\n"

            # Add conversation metadata if available
            if "create_time" in conversation:
                md_content += f"**Created:** {conversation['create_time']}\n\n"

            if "id" in conversation:
                md_content += f"**Conversation ID:** {conversation['id']}\n\n"

            # Add messages if available
            if "mapping" in conversation and conversation["mapping"] is not None:
                mapping = conversation["mapping"]

                # Try to find the conversation structure
                messages = []
                for _, msg_data in mapping.items():
                    if msg_data is None:
                        continue

                    if "message" in msg_data:
                        message = msg_data["message"]
                        if message is None:
                            continue

                        # Use a default timestamp if create_time is missing or None
                        create_time = message.get("create_time", 0)
                        if create_time is None:
                            create_time = 0

                        if "content" in message:
                            if message["content"] is None:
                                continue

                            # Store message with its timestamp for sorting
                            messages.append((create_time, message))

                # Sort messages by timestamp (safely)
                try:
                    messages.sort(key=lambda x: x[0])
                except TypeError:
                    # If sorting fails, just use the original order
                    print(
                        f"Warning: Could not sort messages for {title}, using original order"
                    )

                # Add messages to markdown content
                for _, message in messages:
                    try:
                        if "author" in message and "role" in message["author"]:
                            role = message["author"]["role"]
                            md_content += f"## {role.capitalize()}\n\n"

                        if "content" in message and "parts" in message["content"]:
                            parts = message["content"]["parts"]
                            if parts is not None:
                                for part in parts:
                                    if part:
                                        md_content += f"{part}\n\n"
                    except Exception as e:
                        print(f"Warning: Error processing a message: {e}")
                        continue

            # Write to markdown file
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md_content)

            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error processing conversation {i}: {e}")


def extract_from_titles_file(index=None, search_string=None):
    """Extract conversations based on the titles JSON file.

    Args:
        index (int, optional): If provided, extract the conversation at this index in the titles file.
        search_string (str, optional): If provided, extract conversations with titles containing this string.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    titles_json_file = os.path.join(script_dir, "conversation_titles.json")

    # Check if titles file exists
    if not os.path.exists(titles_json_file):
        print("Titles file not found. Run extract_titles.py first.")
        return

    # Load titles
    with open(titles_json_file, "r", encoding="utf-8") as f:
        titles = json.load(f)

    if index is not None:
        try:
            index = int(index)
            if 0 <= index < len(titles):
                conversation_id = titles[index]["id"]
                print(f"Extracting conversation: {titles[index]['title']}")
                extract_conversations(conversation_id=conversation_id)
            else:
                print(f"Index {index} out of range. Valid range: 0-{len(titles) - 1}")
        except ValueError:
            print(f"Invalid index: {index}. Must be an integer.")

    elif search_string:
        matching_titles = []
        matching_titles.extend(
            title_obj
            for title_obj in titles
            if search_string.lower() in title_obj["title"].lower()
        )
        if matching_titles:
            print(f"Found {len(matching_titles)} matching titles:")
            for i, title_obj in enumerate(matching_titles):
                print(f"{i + 1}. {title_obj['title']}")

            for title_obj in matching_titles:
                extract_conversations(conversation_id=title_obj["id"])
        else:
            print(f"No titles found containing '{search_string}'")

    else:
        print("Please provide either an index or a search string.")


def print_usage():
    """Print usage instructions."""
    print("Usage:")
    print("  1. Extract all conversation titles:")
    print("     python extract_titles.py")
    print()
    print("  2. Extract a specific conversation by title search:")
    print('     python extract_conversations.py --title "Title Search String"')
    print()
    print("  3. Extract a specific conversation by ID:")
    print('     python extract_conversations.py --id "conversation_id"')
    print()
    print("  4. Extract a conversation by index from titles file:")
    print("     python extract_conversations.py --index 5")
    print()
    print("  5. Extract all conversations with titles containing a string:")
    print('     python extract_conversations.py --search "ML"')


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Extract conversations from a specified data folder."
        )
        parser.add_argument(
            "data_folder",
            help="Path to the folder containing the conversations.json file",
        )
        parser.add_argument("--title", help="Title search string", default=None)
        parser.add_argument("--id", help="Specific conversation ID", default=None)
        parser.add_argument(
            "--index", help="Index from titles file", type=int, default=None
        )
        parser.add_argument("--search", help="Search string for titles", default=None)
        args = parser.parse_args()

        if args.index is not None:
            extract_from_titles_file(index=args.index)
        elif args.search:
            extract_from_titles_file(search_string=args.search)
        elif args.title:
            extract_conversations(args.data_folder, search_title=args.title)
        elif args.id:
            extract_conversations(args.data_folder, conversation_id=args.id)
        else:
            print_usage()

        print("Extraction completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
