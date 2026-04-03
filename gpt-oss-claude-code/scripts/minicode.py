#!/usr/bin/env python3
"""
███    ███ ██ ███    ██ ██  ██████  ██████  ██████  ███████
████  ████ ██ ████   ██ ██ ██      ██    ██ ██   ██ ██
██ ████ ██ ██ ██ ██  ██ ██ ██      ██    ██ ██   ██ █████
██  ██  ██ ██ ██  ██ ██ ██ ██      ██    ██ ██   ██ ██
██      ██ ██ ██   ████ ██  ██████  ██████  ██████  ███████

Coding assistant via mlx, transformers, openrouter, or local proxy.

BACKEND env: mlx (default) | transformers | openrouter | local
MODEL env: override model path / id

Key env vars: NANOCODE_WORKSPACE, NANOCODE_UNRESTRICTED_PATHS, MAX_READ_BYTES,
  MAX_READ_LINES, GREP_MAX_MATCHES, BASH_TIMEOUT, MAX_TOOL_OUTPUT_CHARS,
  GLOB_SKIP_DIRS, OPENROUTER_API_KEY, LOCAL_API_KEY, LOCAL_PORT
"""

import contextlib
import glob as globlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request

# Load .env from parent directory
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

BACKEND = os.environ.get("BACKEND", "mlx")

if BACKEND == "openrouter":
    MODEL = os.environ.get("MODEL", "anthropic/claude-3-haiku")
    API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
elif BACKEND == "local":
    MODEL = os.environ.get("MODEL", "gpt-oss-20b")
    API_KEY = os.environ.get("LOCAL_API_KEY", "local")
    LOCAL_PORT = os.environ.get("LOCAL_PORT", "8082")
    API_BASE = f"http://localhost:{LOCAL_PORT}/v1/messages"
elif BACKEND == "transformers":
    MODEL = os.environ.get("MODEL", "deburky/gpt-oss-claude-code")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
else:  # mlx
    MODEL = os.environ.get("MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    from mlx_lm import load
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
MAX_READ_BYTES = int(os.environ.get("MAX_READ_BYTES", str(4 * 1024 * 1024)))
MAX_READ_LINES = int(os.environ.get("MAX_READ_LINES", "800"))
GREP_MAX = int(os.environ.get("GREP_MAX_MATCHES", "80"))
BASH_TIMEOUT = int(os.environ.get("BASH_TIMEOUT", "120"))
MAX_OUT = int(os.environ.get("MAX_TOOL_OUTPUT_CHARS", "48000"))
_GLOB_SKIP = {
    s
    for s in os.environ.get(
        "GLOB_SKIP_DIRS",
        ".git,node_modules,__pycache__,.venv,venv,dist,build,.mypy_cache,.pytest_cache,target",
    ).split(",")
    if s
}

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)


# Path helpers
def workspace_root():
    if w := os.environ.get("NANOCODE_WORKSPACE"):
        return pathlib.Path(w).expanduser().resolve()
    return pathlib.Path(os.getcwd()).resolve()


def paths_unrestricted():
    return os.environ.get("NANOCODE_UNRESTRICTED_PATHS", "").lower() in (
        "1",
        "true",
        "yes",
    )


def resolve_tool_path(raw):
    if not raw or not str(raw).strip():
        raise ValueError("path is required")
    p = pathlib.Path(str(raw).strip()).expanduser()
    root = workspace_root()
    p = p.resolve() if p.is_absolute() else (root / p).resolve()
    if not paths_unrestricted():
        try:
            p.relative_to(root)
        except ValueError:
            raise ValueError(
                f"path {raw!r} resolves outside workspace {root} (set NANOCODE_UNRESTRICTED_PATHS=1)"
            ) from None
    return p


# Tools
def read(args):
    path = resolve_tool_path(args["path"])
    if path.is_dir():
        entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
        return (
            "\n".join(f"  {e.name}{'/' if e.is_dir() else ''}" for e in entries)
            or "(empty)"
        )
    if not path.is_file():
        return f"error: not a file: {path}"
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        return f"error: file too large ({size} bytes, max {MAX_READ_BYTES})"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    offset = int(args.get("offset", 0) or 0)
    if not (0 <= offset <= len(lines)):
        return f"error: offset {offset} out of range (file has {len(lines)} lines)"
    cap = min(
        int(args["limit"]) if args.get("limit") else len(lines) - offset, MAX_READ_LINES
    )
    out = "".join(
        f"{offset + i + 1:4}| {l}" for i, l in enumerate(lines[offset : offset + cap])
    )
    if offset + cap < len(lines):
        out += f"\n... ({len(lines) - offset - cap} more lines; use offset/limit or raise MAX_READ_LINES)"
    return out


def write(args):
    path = resolve_tool_path(args["path"])
    if not confirm(f"Write to {path!r}"):
        return "cancelled"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return "ok"


def edit(args):
    path = resolve_tool_path(args["path"])
    if not path.is_file():
        return f"error: not a file: {path}"
    if path.stat().st_size > MAX_READ_BYTES:
        return f"error: file too large (max {MAX_READ_BYTES} bytes)"
    text = path.read_text(encoding="utf-8", errors="replace")
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times (use all=true)"
    if not confirm(f"Edit {path!r}"):
        return "cancelled"
    path.write_text(
        text.replace(old, new) if args.get("all") else text.replace(old, new, 1),
        encoding="utf-8",
    )
    return "ok"


def glob(args):
    if "pattern" in args and "pat" not in args:
        args["pat"] = args.pop("pattern")
    base = resolve_tool_path(args.get("path", "."))
    if not base.is_dir():
        return f"error: not a directory: {base}"
    files = [
        f
        for f in globlib.glob(str(base / args["pat"]), recursive=True)
        if os.path.isfile(f) and all(p not in _GLOB_SKIP for p in pathlib.Path(f).parts)
    ]
    return "\n".join(sorted(files, key=os.path.getmtime, reverse=True)) or "none"


def grep(args):
    root = resolve_tool_path(args.get("path", "."))
    if not root.is_dir():
        return f"error: grep path must be a directory: {root}"
    rg = shutil.which("rg")
    if not rg:
        return "error: ripgrep (rg) not installed — brew install ripgrep"
    try:
        proc = subprocess.run(
            [rg, "-n", "--color", "never", "--no-heading", "-e", args["pat"], "."],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "error: ripgrep timed out (90s)"
    if proc.returncode not in (0, 1):
        return (
            f"error: ripgrep failed ({proc.returncode}): {(proc.stderr or '').strip()}"
        )
    raw = proc.stdout.splitlines()
    body = "\n".join(raw[:GREP_MAX]) or "none"
    if len(raw) > GREP_MAX:
        body += f"\n... ({len(raw) - GREP_MAX} more; raise GREP_MAX_MATCHES)"
    return body


def confirm(prompt):
    return input(f"\n{YELLOW}⚠ {prompt} [y/N]{RESET} ").strip().lower() in ("y", "yes")


def bash(args):
    if not confirm(f"Run: {args['cmd']!r}"):
        return "cancelled"
    proc = subprocess.Popen(
        args["cmd"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.getcwd(),
    )
    output_lines: list[str] = []

    def reader():
        with contextlib.suppress(Exception):
            assert proc.stdout is not None
            for line in proc.stdout:
                output_lines.append(line)
                print(f"{DIM}│ {line.rstrip()}{RESET}", flush=True)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    deadline = time.monotonic() + BASH_TIMEOUT
    timed_out = False
    while proc.poll() is None:
        if time.monotonic() > deadline:
            timed_out = True
            proc.kill()
            output_lines.append(f"\n(timed out after {BASH_TIMEOUT}s)\n")
            break
        time.sleep(0.05)
    t.join(timeout=2.0)
    if not timed_out:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2.0)
    return "".join(output_lines).strip() or "(empty)"


TOOLS = {
    "read": (
        "Read file with line numbers, or list directory",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": ("Write content to file", {"path": "string", "content": "string"}, write),
    "edit": (
        "Replace old with new in file",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex (requires rg)",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "bash": ("Run shell command", {"cmd": "string"}, bash),
}


def run_tool(name, args):
    try:
        result = TOOLS[name][2](args)
        if len(result) > MAX_OUT:
            result = (
                result[:MAX_OUT]
                + f"\n... [truncated {len(result) - MAX_OUT} chars; raise MAX_TOOL_OUTPUT_CHARS]"
            )
        return result
    except Exception as e:
        return f"error: {e}"


# Inference
def flatten_content(content):
    """Flatten Anthropic-style content list to plain string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "tool_use":
            parts.append(
                f'<tool_call>{{"tool": "{block["name"]}", "args": {json.dumps(block["input"])}}}</tool_call>'
            )
        elif block.get("type") == "tool_result":
            parts.append(f"Tool result: {block.get('content', '')}")
    return "\n".join(parts)


def strip_gptoss_tokens(text):
    if "<|channel|>final<|message|>" in text:
        text = text.split("<|channel|>final<|message|>")[-1]
    return re.sub(r"<\|[^>]+\|>", "", text).strip()


def truncate_at_turn_leak(text):
    return next(
        (
            text.split(m)[0].strip()
            for m in ("\nUser:", "\nSystem:", "\nHuman:", "\n\nUser:", "\n\nSystem:")
            if m in text
        ),
        text,
    )


def _tool_call_complete(text):
    """Return end index of first complete <tool_call> block, or -1."""
    start = text.find("<tool_call>")
    if start == -1:
        return -1
    end_tag = text.find("</tool_call>", start)
    if end_tag != -1:
        return end_tag + len("</tool_call>")
    brace_start = text.find("{", start)
    if brace_start == -1:
        return -1
    depth, last = 0, -1
    for i, ch in enumerate(text[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last = i + 1
                break
    return last


def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        with contextlib.suppress(Exception):
            d = json.loads(m.group(1))
            if d.get("tool") in TOOLS:
                calls.append(
                    {
                        "type": "tool_use",
                        "id": f"call_{len(calls)}",
                        "name": d["tool"],
                        "input": d.get("args", {}),
                    }
                )
    return calls


def _http_post(url, payload, headers):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise Exception(f"HTTP {e.code}: {e.read().decode()}") from e


def get_response(messages, system_prompt, mlx_state):
    flat = [
        {"role": m["role"], "content": flatten_content(m["content"])} for m in messages
    ]

    if BACKEND == "openrouter":
        data = _http_post(
            "https://openrouter.ai/api/v1/chat/completions",
            {
                "model": MODEL,
                "messages": [{"role": "system", "content": system_prompt}] + flat,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.3,
            },
            {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        )
        return data["choices"][0]["message"]["content"]

    if BACKEND == "local":
        data = _http_post(
            API_BASE,
            {
                "model": MODEL,
                "system": system_prompt,
                "messages": flat,
                "max_tokens": MAX_TOKENS,
            },
            {
                "Content-Type": "application/json",
                "x-api-key": API_KEY,
                "anthropic-version": "2023-06-01",
            },
        )
        text = "".join(
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        )
        return strip_gptoss_tokens(text)

    if BACKEND == "transformers":
        model, tokenizer = mlx_state
        inputs = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}] + flat,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=MAX_TOKENS, temperature=0.3, do_sample=True
            )
        raw = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False
        )
        end = _tool_call_complete(raw)
        if end != -1:
            raw = raw[:end]
        return truncate_at_turn_leak(strip_gptoss_tokens(raw))

    # mlx
    model, tokenizer = mlx_state
    chat = [{"role": "system", "content": system_prompt}]
    for m in messages:
        if c := flatten_content(m["content"]):
            chat.append({"role": m["role"], "content": c})
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=0.3, top_p=0.95, min_p=0.0, min_tokens_to_keep=1)
    out = ""
    for chunk in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler
    ):
        out += chunk.text
        end = _tool_call_complete(out)
        if end != -1:
            out = out[:end]
            break
    if out.startswith(prompt):
        out = out[len(prompt) :].strip()
    return truncate_at_turn_leak(strip_gptoss_tokens(out))


# History

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".minicode_history.json"
)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with contextlib.suppress(Exception):
            with open(HISTORY_FILE) as f:
                return json.load(f)
    return []


def save_history(messages):
    with contextlib.suppress(Exception):
        with open(HISTORY_FILE, "w") as f:
            json.dump(messages, f)


def compact_messages(messages, model, tokenizer, system_prompt):
    """Summarize conversation history to reduce context length (mlx only)."""
    if not messages:
        return messages
    history_text = "".join(
        f"{m['role']}: {flatten_content(m['content'])}\n" for m in messages
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Summarize this conversation in 3-5 bullet points:\n\n{history_text}",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampler = make_sampler(temp=0.3, top_p=0.95, min_p=0.0, min_tokens_to_keep=1)
    summary = "".join(
        c.text
        for c in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=512, sampler=sampler
        )
    )
    if summary.startswith(prompt):
        summary = summary[len(prompt) :].strip()
    return [
        {"role": "user", "content": f"[Conversation summary]\n{summary}"},
        {
            "role": "assistant",
            "content": "Understood, I have the context from the summary.",
        },
    ]


def git_context():
    with contextlib.suppress(Exception):
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode == 0 and r.stdout.strip():
            return f"\nGit status:\n{r.stdout.strip()}"
    return ""


def render_markdown(text):
    return re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)


def load_model():
    """Load model for the current backend and return mlx_state (or None for API backends)."""
    if BACKEND == "mlx":
        print(f"{YELLOW}Loading model...{RESET}")
        model, tokenizer = load(MODEL)
        print(f"{GREEN}✓ Loaded: {getattr(model, 'name', MODEL)}{RESET}\n")
        return (model, tokenizer)
    if BACKEND == "transformers":
        print(f"{YELLOW}Loading model via transformers...{RESET}")
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _mdl = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map=_device
        )
        print(f"{GREEN}✓ Loaded on {_device}: {MODEL}{RESET}\n")
        return (_mdl, _tok)
    if BACKEND == "local":
        print(f"{DIM}Local proxy at {API_BASE}{RESET}\n")
    elif BACKEND == "openrouter":
        if not API_KEY:
            print(f"{RED}⏺ OPENROUTER_API_KEY not set{RESET}")
            raise SystemExit(1)
        print(f"{DIM}OpenRouter ({MODEL}){RESET}\n")
    return None


def build_system_prompt():
    """Build the system prompt with workspace context and tool definitions."""
    ws = workspace_root()
    path_rule = (
        "Paths are not restricted to the workspace."
        if paths_unrestricted()
        else "Relative paths resolve under the workspace. Absolute paths must stay inside it."
    )
    return f"""You are a helpful coding assistant with tools to interact with the file system.
Workspace root: {ws}
Process cwd: {os.getcwd()}
{path_rule}{git_context()}

IMPORTANT: You MUST use tools by formatting them exactly as shown below.

Available tools:
- read(path, offset, limit): Read a file or list a directory
- write(path, content): Write to a file
- edit(path, old, new): Replace text in a file (old must be unique unless all=true)
- glob(pat): Find files matching pattern
- grep(pat): Search for text in files
- bash(cmd): Run a shell command

To use a tool, format it EXACTLY like this:
<tool_call>{{"tool": "name", "args": {{"key": "value"}}}}</tool_call>

Examples:
<tool_call>{{"tool": "read", "args": {{"path": "file.py", "offset": 0, "limit": 20}}}}</tool_call>
<tool_call>{{"tool": "glob", "args": {{"pat": "*.py"}}}}</tool_call>

When reading a file, always pass offset and limit. When you finish a task, summarize what you changed.
CRITICAL: You MUST use tools for file operations. Never say you can't access files!"""


def run_agent_turn(messages, system_prompt, mlx_state):
    """Generate a response and execute any tool calls, repeating until no tools remain."""
    while True:
        print(f"{DIM}Generating...{RESET}", end="\r", flush=True)
        response_text = get_response(messages, system_prompt, mlx_state)
        print(" " * 20, end="\r")

        tool_calls = parse_tool_calls(response_text)
        display_text = re.sub(
            r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL
        ).strip()

        if display_text:
            print(f"\n{CYAN}⏺{RESET} {render_markdown(display_text)}")

        content_blocks = (
            [{"type": "text", "text": display_text}] if display_text else []
        )
        tool_results = []
        for tc in tool_calls:
            arg_preview = str(list(tc["input"].values())[0])[:50] if tc["input"] else ""
            print(
                f"\n{GREEN}⏺ {tc['name'].capitalize()}{RESET}({DIM}{arg_preview}{RESET})"
            )
            result = run_tool(tc["name"], tc["input"])
            lines = result.split("\n")
            preview = lines[0][:60] + (
                f" ... +{len(lines) - 1} lines"
                if len(lines) > 1
                else ("..." if len(lines[0]) > 60 else "")
            )
            print(f"{DIM}⎿ {preview}{RESET}")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tc["id"], "content": result}
            )
            content_blocks.append(tc)

        messages.append({"role": "assistant", "content": content_blocks})
        if not tool_results:
            break
        messages.append({"role": "user", "content": tool_results})


def handle_slash_command(cmd, messages, mlx_state, system_prompt):
    """Handle a slash command. Returns 'quit', 'handled', or None if not a command."""
    if cmd in ("/q", "exit"):
        save_history(messages)
        return "quit"
    if cmd == "/c":
        messages.clear()
        save_history(messages)
        print(f"{GREEN}⏺ Cleared{RESET}")
        return "handled"
    if cmd == "/compact":
        if BACKEND == "mlx" and mlx_state:
            print(f"{DIM}Compacting history...{RESET}")
            model, tokenizer = mlx_state
            messages[:] = compact_messages(messages, model, tokenizer, system_prompt)
            save_history(messages)
            print(f"{GREEN}⏺ Compacted to {len(messages)} messages{RESET}")
        else:
            print(f"{YELLOW}⏺ /compact only available in mlx backend{RESET}")
        return "handled"
    if cmd == "/help":
        print(f"{DIM}/c — clear  /compact — summarize history (mlx)  /q — quit{RESET}")
        return "handled"
    return None


def main():
    os.environ.setdefault("NANOCODE_WORKSPACE", str(pathlib.Path.cwd().resolve()))
    print(f"{BOLD}minicode{RESET} | {DIM}{BACKEND}:{MODEL}{RESET}\n")
    mlx_state = load_model()
    system_prompt = build_system_prompt()
    messages = load_history()
    if messages:
        print(f"{DIM}⏺ Restored {len(messages)} messages{RESET}\n")

    while True:
        try:
            user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            if not user_input:
                continue
            action = handle_slash_command(
                user_input, messages, mlx_state, system_prompt
            )
            if action == "quit":
                break
            if action == "handled":
                continue
            messages.append({"role": "user", "content": user_input})
            run_agent_turn(messages, system_prompt, mlx_state)
            save_history(messages)
        except KeyboardInterrupt:
            save_history(messages)
            print(f"\n{YELLOW}⏺ Interrupted{RESET}")
            break
        except EOFError:
            break
        except Exception as err:
            print(f"{RED}⏺ Error: {err}{RESET}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
