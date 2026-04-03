"""
Extract tool-use training examples from Claude Code conversation history.

Reads ~/.claude/projects/**/*.jsonl, reconstructs conversation threads,
filters for sequences with real tool calls, and outputs Anthropic-format
messages ready for prepare_dataset.py.

Usage:
    python scripts/extract_training_data.py [--out custom-data/extracted.jsonl]
"""

import argparse
import json
import glob
import random
from pathlib import Path

TARGET_TOOLS = {"Bash", "Read", "Write", "Edit", "Glob", "Grep"}
MAX_TURNS = 12       # max messages per example (user+assistant combined)
MIN_TOOL_CALLS = 1   # skip threads without at least this many tool_use turns
MAX_RESULT_LEN = 800 # truncate tool results to keep prompts short


def clean_tool_result_content(inner) -> str:
    """Extract text from tool_result content, return empty string if no text."""
    if isinstance(inner, str):
        return inner[:MAX_RESULT_LEN]
    if isinstance(inner, list):
        parts = [x.get("text", "") for x in inner if x.get("type") == "text"]
        return "\n".join(parts)[:MAX_RESULT_LEN]
    return ""


def thread_to_messages(thread: list) -> list:
    """
    Convert a list of JSONL records (user/assistant) into clean training messages.
    Tool blocks are serialized as JSON strings to match the bridge.py format.
    """
    messages = []

    for record in thread:
        rtype = record.get("type")
        content = record.get("message", {}).get("content", "")

        if rtype == "user":
            if isinstance(content, str):
                text = content.strip()
                if text:
                    messages.append({"role": "user", "content": text})
            elif isinstance(content, list):
                tool_results = []
                text_parts = []
                for block in content:
                    btype = block.get("type")
                    if btype == "tool_result":
                        text = clean_tool_result_content(block.get("content", []))
                        if text:
                            tool_results.append({
                                "type": "tool_result",
                                "content": text,
                            })
                    elif btype == "text":
                        text_parts.append(block.get("text", ""))

                if tool_results:
                    messages.append({
                        "role": "user",
                        "content": json.dumps(tool_results),
                    })
                elif text_parts:
                    text = " ".join(text_parts).strip()
                    if text:
                        messages.append({"role": "user", "content": text})

        elif rtype == "assistant":
            if isinstance(content, list):
                tool_blocks = []
                text_parts = []
                for block in content:
                    btype = block.get("type")
                    if btype == "tool_use" and block.get("name") in TARGET_TOOLS:
                        tool_blocks.append({
                            "type": "tool_use",
                            "name": block["name"],
                            "input": block.get("input", {}),
                        })
                    elif btype == "text":
                        text = block.get("text", "").strip()
                        if text:
                            text_parts.append(text)
                    # skip "thinking" blocks

                if tool_blocks:
                    # Tool call: serialize as JSON string (matches bridge.py format)
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(tool_blocks),
                    })
                elif text_parts:
                    messages.append({
                        "role": "assistant",
                        "content": "\n".join(text_parts),
                    })

    return messages


def is_valid(messages: list) -> bool:
    """Require alternating roles, tool calls present, ends with assistant."""
    if len(messages) < 2:
        return False
    if messages[0]["role"] != "user":
        return False
    # First user message must be plain text, not a tool_result
    if messages[0]["content"].startswith("["):
        return False
    if messages[-1]["role"] != "assistant":
        return False

    # Count tool_use turns
    tool_turns = sum(
        1 for m in messages
        if m["role"] == "assistant" and m["content"].startswith("[")
    )
    if tool_turns < MIN_TOOL_CALLS:
        return False

    # Check alternating roles (no consecutive same role)
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            return False

    return True


def build_threads(records: list) -> list[list]:
    """Reconstruct linear conversation threads from parentUuid links."""
    by_uuid = {}
    has_children = set()

    for r in records:
        uid = r.get("uuid")
        if uid and r.get("type") in ("user", "assistant"):
            by_uuid[uid] = r
            parent = r.get("parentUuid")
            if parent:
                has_children.add(parent)

    # Leaves = nodes with no children in this file
    leaves = [uid for uid in by_uuid if uid not in has_children]

    threads = []
    seen = set()

    for leaf in leaves:
        thread = []
        cur = leaf
        for _ in range(60):
            r = by_uuid.get(cur)
            if not r:
                break
            thread.insert(0, r)
            cur = r.get("parentUuid")
            if not cur:
                break

        sig = tuple(r.get("uuid") for r in thread)
        if sig in seen:
            continue
        seen.add(sig)

        # Only keep user/assistant records
        thread = [r for r in thread if r.get("type") in ("user", "assistant")]
        if thread:
            threads.append(thread)

    return threads


def sliding_windows(messages: list, window: int = MAX_TURNS) -> list[list]:
    """
    Yield overlapping windows of `window` messages, each starting with user
    and ending with assistant, and containing at least one tool call.
    """
    windows = []
    for start in range(0, len(messages) - 1, 2):
        end = min(start + window, len(messages))
        chunk = messages[start:end]
        # Ensure ends on assistant
        if chunk[-1]["role"] != "assistant":
            chunk = chunk[:-1]
        if is_valid(chunk):
            windows.append(chunk)
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-dir",
        nargs="+",
        default=[
            str(Path.home() / ".claude" / "projects"),
            str(Path(__file__).parent.parent / "claude-projects"),
        ],
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent.parent / "custom-data" / "extracted.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=500)
    args = parser.parse_args()

    files = []
    for d in args.history_dir:
        files += [str(p) for p in Path(d).rglob("*.jsonl")]
    print(f"Found {len(files)} conversation files")

    all_examples = []

    for path in files:
        try:
            with open(path) as f:
                records = [json.loads(line) for line in f]
        except Exception as e:
            print(f"  Skip {path}: {e}")
            continue

        threads = build_threads(records)
        for thread in threads:
            messages = thread_to_messages(thread)
            if not messages:
                continue

            # Try the full thread first, then sliding windows for long ones
            if len(messages) <= MAX_TURNS and is_valid(messages):
                all_examples.append({"messages": messages})
            else:
                for window in sliding_windows(messages):
                    all_examples.append({"messages": window})

    print(f"Extracted {len(all_examples)} raw examples")

    # Deduplicate by first user message
    seen_first = set()
    deduped = []
    for ex in all_examples:
        key = ex["messages"][0]["content"][:100]
        if key not in seen_first:
            seen_first.add(key)
            deduped.append(ex)

    random.seed(args.seed)
    random.shuffle(deduped)
    examples = deduped[: args.max_examples]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(examples)} examples → {out}")

    # Preview
    if examples:
        ex = examples[0]
        print(f"\nSample ({len(ex['messages'])} turns):")
        for m in ex["messages"][:4]:
            preview = m["content"][:120].replace("\n", " ")
            print(f"  [{m['role']}] {preview}")


if __name__ == "__main__":
    main()
