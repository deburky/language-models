from dotenv import load_dotenv
from openai import OpenAI
import os
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = "gpt-3.5-turbo"
# Call the openai ChatCompletion endpoint
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "What is Flask used for?"}],
    max_tokens=100,
    temperature=1.2,
)

# Print response beautifully
console = Console()
md = Markdown(response.choices[0].message.content)
console.print(Panel(md, title=f"Response from {model}", style="bold cyan1"))
