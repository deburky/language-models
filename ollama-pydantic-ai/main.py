import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.exceptions import UnexpectedModelBehavior
import uvicorn

load_dotenv()

# === Pydantic Models ===

class Subtask(BaseModel):
    name: str
    description: str
    python_code: str = ""

class InputTask(BaseModel):
    description: str

class SubtasksList(BaseModel):
    tasks: List[Subtask]

class GeneratedFunction(BaseModel):
    task_name: str
    python_code: str

class FullPipelineOutput(BaseModel):
    tasks: List[Subtask]

# === Model Setup ===

ollama_model = OpenAIModel(
    model_name="llama3.1:8b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "ollama")
)

extract_agent = Agent(model=ollama_model, result_type=SubtasksList)
codegen_agent = Agent(model=ollama_model, result_type=GeneratedFunction)

# === FastAPI App ===

app = FastAPI(
    title="AI Pipeline Assistant",
    description="Extract subtasks and generate Python code for each.",
    version="0.2.0",
    default_response_class=ORJSONResponse,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# === API Endpoint ===

@app.post("/pipeline", response_model=FullPipelineOutput)
async def generate_pipeline(input: InputTask, request: Request):
    try:
        result = await extract_agent.run(
            f"""
You are a task extraction assistant.

Please respond only with JSON in the format:
{{
  "tasks": [
    {{"name": "Task Name", "description": "Task Description"}},
    ...
  ]
}}

Description:
{input.description}
""",
            model_settings={"max_retries": 5},
        )
    except UnexpectedModelBehavior:
        print("Failed to extract subtasks.")
        print(extract_agent.last_run_messages)
        raise

    enriched_tasks = []

    for task in result.data.tasks:
        try:
            code_prompt = (
                f"Write a Python function for the task '{task.name}': {task.description}. "
                "The function should be realistic and self-contained."
            )
            code_result = await codegen_agent.run(code_prompt)
            task.python_code = code_result.data.python_code
        except UnexpectedModelBehavior:
            print(f"Codegen failed for: {task.name}")
            print(codegen_agent.last_run_messages)
            task.python_code = "# Failed to generate code"

        enriched_tasks.append(task)

    return {"tasks": enriched_tasks}

# === Entrypoint ===

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
