from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(
    base_url="https://api.deepseek.com/v1", api_key=os.getenv("DEEPSEEK_API_KEY")
)

models = ["deepseek-chat", "deepseek-reasoner"]
model = models[0]
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain what Flask does?"}],
    max_tokens=250,
)

console = Console()
md = Markdown(response.choices[0].message.content)
console.print(Panel(md, title=f"Response from {model}", style="bold cyan1"))
