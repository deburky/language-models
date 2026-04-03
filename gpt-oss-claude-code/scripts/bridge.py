"""
Minimal Anthropic /v1/messages → mlx_lm /v1/chat/completions bridge.
Stdlib only. Handles streaming and non-streaming.

Usage (called by serve.sh):
    python bridge.py --mlx-port 8080 --port 8082
"""

import argparse
import contextlib
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_PATH = _SCRIPT_DIR / "model-finetuned-q4"

MLX_URL = "http://localhost:8080"
MODEL_PATH = str(_DEFAULT_MODEL_PATH)
_MLX_LOCK = threading.Lock()  # serialize requests to mlx_lm


def strip_cot(text: str) -> str:
    """Extract only the final channel content from a harmony response."""
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>")[-1]
    for stop in ("<|return|>", "<|end|>", "<|start|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def parse_tool_use(text: str):
    """Return list of tool_use dicts if text is a tool_use JSON array, else None."""
    text = text.strip()
    if not text.startswith("["):
        return None
    with contextlib.suppress(Exception):
        data = json.loads(text)
        if isinstance(data, list) and data and data[0].get("type") == "tool_use":
            return data
    return None


HARMONY_PREFIX = (
    "Reasoning: low\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message.\n\n"
    "To call a tool, output a JSON array in the final channel.\n"
    "Example: [{\"type\": \"tool_use\", \"name\": \"Bash\", \"input\": {\"command\": \"ls\"}}]\n\n"
    "Available tools: Bash (command), Read (file_path), Write (file_path + content), "
    "Edit (file_path + old_string + new_string), Glob (pattern + path), Grep (pattern + path).\n\n"
    "After receiving a tool_result, call another tool or give your final answer as plain text.\n"
    "Only use tools when you need to read files, run commands, or search code. "
    "Answer factual and conceptual questions directly without tools."
)

FALLBACK_SYSTEM = "You are a concise coding assistant with tool access."


def build_system(tools: list, client_system="") -> str:
    if isinstance(client_system, list):
        client_system = " ".join(b.get("text", "") for b in client_system if b.get("type") == "text")
    base = client_system.strip() if client_system else FALLBACK_SYSTEM
    return f"{base}\n\n{HARMONY_PREFIX}"


MAX_HISTORY = 4  # keep last N user/assistant turns to limit prompt size


def to_openai(body: dict) -> dict:
    tools = body.get("tools", [])
    messages = [{"role": "system", "content": build_system(tools, body.get("system", ""))}]
    raw_messages = body.get("messages", [])[-MAX_HISTORY:]
    for m in raw_messages:
        content = m["content"]
        if isinstance(content, list):
            types = {b.get("type") for b in content}
            if "tool_result" in types or "tool_use" in types:
                # Serialize structured blocks as JSON string the model was trained on
                content = json.dumps(content)
            else:
                content = "".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
        # Strip any leaked harmony channel tags from history messages
        if isinstance(content, str) and "<|channel|>" in content:
            content = strip_cot(content)
        msg = {"role": m["role"], "content": content}
        # For assistant history, split analysis into thinking field so tokenizer accepts it
        if m["role"] == "assistant" and isinstance(content, str):
            msg["content"] = content
        messages.append(msg)
    req = {
        "model": MLX_URL.replace("http://", ""),
        "messages": messages,
        "stream": body.get("stream", False),
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 1.0),
        "stop": body.get("stop", []),
    }
    return req


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        import sys

        print(f"[bridge] {self.command} {self.path}", file=sys.stderr, flush=True)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            # Return all Claude model names so Claude Code finds whichever it checks for
            model_ids = [
                "claude-sonnet-4-6",
                "claude-sonnet-4-5",
                "claude-opus-4-5",
                "claude-haiku-4-5",
                "claude-haiku-4-5-20251001",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "gpt-oss-20b",
                "local",
            ]
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": m,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                        }
                        for m in model_ids
                    ],
                },
            )
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/messages"):
            self._json(404, {"error": "not found"})
            return
        try:
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
        except Exception as e:
            self._json(400, {"error": str(e)})
            return

        # Log tool definitions Claude Code sends (first request only)
        if body.get("tools"):
            import sys

            names = [t.get("name") for t in body["tools"]]
            print(f"[bridge] tools from client: {names}", file=sys.stderr, flush=True)

        oai = to_openai(body)
        oai["model"] = (
            MODEL_PATH  # always route to local model regardless of requested name
        )

        try:
            req = Request(
                f"{MLX_URL}/v1/chat/completions",
                json.dumps(oai).encode(),
                {"Content-Type": "application/json"},
            )
            with _MLX_LOCK:
                if body.get("stream"):
                    self._stream(req, body.get("model", "gpt-oss-20b"))
                else:
                    self._complete(req, body.get("model", "gpt-oss-20b"))
        except URLError as e:
            self._json(503, {"error": f"Cannot reach mlx_lm: {e}"})

    def _complete(self, req, model):
        resp = json.loads(urlopen(req).read())
        msg = resp["choices"][0]["message"]
        raw = msg.get("content") or ""
        # vllm-mlx converts harmony JSON arrays to OpenAI tool_calls in non-streaming mode
        # (content becomes null). Reconstruct the harmony tool call JSON from tool_calls.
        if not raw and msg.get("tool_calls"):
            tool_blocks = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                tool_blocks.append({"type": "tool_use", "name": name, "input": args})
            raw = json.dumps(tool_blocks)
        text = strip_cot(raw)
        if not text and "<|message|>" in raw:
            text = raw.split("<|message|>")[-1].strip()
            for stop in ("<|return|>", "<|end|>", "<|start|>"):
                if stop in text:
                    text = text.split(stop)[0]
            text = text.strip()
        usage = resp.get("usage", {})

        if tool_blocks := parse_tool_use(text):
            content = [
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": b["name"],
                    "input": b.get("input", {}),
                }
                for b in tool_blocks
                if b.get("type") == "tool_use"
            ]
            stop_reason = "tool_use"
        else:
            content = [{"type": "text", "text": text}]
            stop_reason = "end_turn"

        self._json(
            200,
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "content": content,
                "model": model,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
            },
        )

    def _stream(self, req, model):
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        def emit(event, data):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        emit("ping", {"type": "ping"})

        buf = ""
        in_final = False
        mode = None  # "text" or "tool" — determined from first non-whitespace char
        stop_reason = "end_turn"

        try:
            with urlopen(req) as resp:
                for raw in resp:
                    line = raw.decode().rstrip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except Exception:
                        continue

                    text = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if not text:
                        continue
                    buf += text

                    # Suppress output until we reach the final channel
                    if not in_final:
                        if "<|channel|>final<|message|>" in buf:
                            in_final = True
                            buf = buf.split("<|channel|>final<|message|>")[-1]
                        else:
                            continue

                    # Stop on harmony end tokens
                    done = False
                    for stop_tok in ("<|return|>", "<|end|>", "<|start|>"):
                        if stop_tok in buf:
                            buf = buf.split(stop_tok)[0]
                            done = True
                            break

                    # Determine mode from first non-whitespace content
                    if mode is None and buf.strip():
                        mode = "tool" if buf.lstrip().startswith("[") else "text"
                        if mode == "text":
                            emit(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )

                    if mode == "text" and buf:
                        emit(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": buf},
                            },
                        )
                        buf = ""
                    # mode == "tool": keep buf accumulating

                    if done:
                        break

        except BrokenPipeError:
            return

        # Fallback: stream ended without seeing final channel (hit max_tokens in CoT)
        if not in_final and buf:
            buf = strip_cot(buf)
            # strip_cot may return text still prefixed with channel headers if no final channel
            if "<|message|>" in buf:
                buf = buf.split("<|message|>")[-1].strip()
            if buf:
                mode = "text"
                emit(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": buf},
                    },
                )

        # Emit tool_use blocks collected in buf
        if mode == "tool":
            tool_blocks = parse_tool_use(buf.strip())
            if tool_blocks:
                stop_reason = "tool_use"
                for i, block in enumerate(
                    b for b in tool_blocks if b.get("type") == "tool_use"
                ):
                    emit(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": i,
                            "content_block": {
                                "type": "tool_use",
                                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": block["name"],
                                "input": {},
                            },
                        },
                    )
                    emit(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(block.get("input", {})),
                            },
                        },
                    )
                    emit(
                        "content_block_stop", {"type": "content_block_stop", "index": i}
                    )
            else:
                # Malformed JSON — fall back to text
                mode = "text"
                emit(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                emit(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": buf.strip()},
                    },
                )
                emit("content_block_stop", {"type": "content_block_stop", "index": 0})

        if mode == "text":
            emit("content_block_stop", {"type": "content_block_stop", "index": 0})

        emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        )
        emit("message_stop", {"type": "message_stop"})

    def _json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-port", type=int, default=8080)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--model-path", default=str(_DEFAULT_MODEL_PATH))
    args = parser.parse_args()
    MLX_URL = f"http://localhost:{args.mlx_port}"
    MODEL_PATH = args.model_path
    print(f"Bridge: /v1/messages :{args.port} → mlx_lm :{args.mlx_port} ({MODEL_PATH})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
