from openai import OpenAI
from io import BytesIO
import pandas as pd
from rich.table import Table
from rich.live import Live
import time

# Helper function to convert tables
def pyarrow_to_csv_buffer(table):
    """PyArrow table to CSV buffer."""
    csv_buffer = BytesIO()
    df = table.to_pandas()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer

def delete_all_assistants_and_files():
    """Deletes all OpenAI assistants and files."""
    client = OpenAI()

    # Delete all assistants
    assistants = client.beta.assistants.list()
    if assistants.data:
        print(f"🔍 Found {len(assistants.data)} assistants to delete...")
        for assistant in assistants.data:
            print(f"🗑️ Deleting Assistant: {assistant.name} (ID: {assistant.id})")
            client.beta.assistants.delete(assistant.id)
        print("✅ All assistants deleted!")
    else:
        print("✅ No assistants found.")

    # Delete all files
    files = client.files.list()
    if files.data:
        print(f"🔍 Found {len(files.data)} files to delete...")
        for file in files.data:
            print(f"🗑️ Deleting File: {file.filename} (ID: {file.id})")
            client.files.delete(file.id)
        print("✅ All files deleted!")
    else:
        print("✅ No files found.")



# Single conversation table for Rich Live
def fetch_messages(client, thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    table = Table(title="AI Underwriting Conversation (Live)", header_style="cyan bold")
    table.add_column("Role", width=12)
    table.add_column("Message", overflow="fold")
    content = "[No content]"
    for message in reversed(messages.data):
        if message.content:
            block = message.content[0]
            if block.type == "file":
                content_text = "[File received]"
            elif block.type == "image_file":
                content_text = "[Image received]"
            elif block.type == "text":
                content_text = block.text.value.strip()
                text = content_text
        else:
            text = "[No content]"
        table.add_row(message.role, content_text)
    return table


# Run assistant
class AssistantRunner:
    def __init__(self, client):
        self.client = client

    def start(self, assistant_id, thread_id, user_instructions):
        """Create user message and start assistant run, return run object."""
        self.client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=user_instructions
        )
        return self.client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=assistant_id
        )
