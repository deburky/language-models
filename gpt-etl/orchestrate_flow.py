import json
import os
import subprocess

from prefect import flow, task

DATA_FOLDER = "3fdedecf56a7ea489b64ea9493fea222c1636c891dd604bea8a7cde36ad6324e-2025-04-29-11-50-40"
CONVERSATIONS_DIR = "conversations"
CONVERSATIONS_JSON_PATH = "3fdedecf56a7ea489b64ea9493fea222c1636c891dd604bea8a7cde36ad6324e-2025-04-29-11-50-40/conversations.json"

@task
def extract_titles():
    """Task to extract titles."""
    subprocess.run(["uv", "run", "extract_titles.py", DATA_FOLDER], check=True)

@task
def extract_conversations():
    """Task to extract conversations."""
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    with open(CONVERSATIONS_JSON_PATH, "r") as file:
        conversations = json.load(file)
        for conversation in conversations:
            title = conversation["title"]
            file_path = os.path.join(CONVERSATIONS_DIR, f"{title}.md")
            
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
            with open(file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)

@task
def etl_to_db():
    """Task to embed and store data in Chroma DB."""
    subprocess.run(["uv", "run", "etl_to_db.py"], check=True)

@task
def search_db():
    """Task to search the Chroma DB."""
    subprocess.run(["uv", "run", "search_db.py"], check=True)

@flow(name="Chroma DB Orchestration")
def chroma_db_flow():
    titles_task = extract_titles()
    conversations_task = extract_conversations()
    etl_task = etl_to_db(wait_for=[titles_task, conversations_task])
    search_task = search_db(wait_for=[etl_task])

if __name__ == "__main__":
    chroma_db_flow()
