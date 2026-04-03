"""
Convert custom-data/*.jsonl → mlx-lm LoRA training format.

Each line of output is {"text": "<full harmony-rendered conversation>"} so mlx-lm does not
re-apply the chat template on top of <|channel|> tokens.

Output: data/train.jsonl + data/valid.jsonl (80/20 split)

Rows whose first user message looks like a mid-conversation ack ("ok", "yes", "now what?", …)
are dropped. Add clean single-turn Q&A in custom-data (see anthropic_style_finetune_singleturn_qa.jsonl)
and no-tool factual rows in anthropic_style_finetune_tools_negs.jsonl.

Usage:
    python prepare_dataset.py [--model openai/gpt-oss-20b] [--input-dir ../custom-data]
"""

import argparse
import json
import random
import re
from pathlib import Path

import tiktoken

ENCODING = tiktoken.get_encoding("o200k_base")

MAX_TOKENS = 1024  # hard cap using tiktoken o200k_base
MAX_TOOL_CALLS = 2  # allow up to 2 tool calls per example
MAX_MESSAGES = 7    # user + assistant(tool) + user(result) + assistant(tool2) + user(result2) + assistant(final)

# First user turns like "ok" / "now what?" teach mid-conversation behavior; drop them.
_EXACT_FIRST_USER_FOLLOWUPS = frozenset(
    {
        "ok",
        "okay",
        "k",
        "yes",
        "yeah",
        "yep",
        "yup",
        "no",
        "nope",
        "nah",
        "correct",
        "right",
        "thanks",
        "thank you",
        "thx",
        "ty",
        "got it",
        "cool",
        "nice",
        "great",
        "perfect",
        "continue",
        "next",
        "more",
        "please",
        "pls",
        "sure",
        "hmm",
        "hm",
        "interesting",
        "exactly",
        "i see",
        "understood",
        "alright",
        "all right",
        "do it",
        "try again",
        "again",
        "go ahead",
        "proceed",
        "mhm",
        "indeed",
        "sounds good",
        "agreed",
    }
)

_FIRST_USER_FOLLOWUP_RE = re.compile(
    r"^(?:"
    r"ok(?:ay)?|yes|yeah|yep|no|nope|correct|right|thanks?|thx|sure|got it"
    r")[\s,!.]*$",
    re.IGNORECASE,
)

_FIRST_USER_FOLLOWUP_PHRASE_START = (
    "now what",
    "what now",
    "what next",
    "and now",
    "so what",
    "what about now",
    "ok so what",
    "yes what",
    "ok next",
)


def _normalize_first_user_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def is_followup_first_user(content: str) -> bool:
    """True if the first user message looks like a mid-thread ack, not a standalone request."""
    t = _normalize_first_user_text(content)
    if not t:
        return True
    if t in _EXACT_FIRST_USER_FOLLOWUPS:
        return True
    if _FIRST_USER_FOLLOWUP_RE.match(t):
        return True
    for prefix in _FIRST_USER_FOLLOWUP_PHRASE_START:
        if t == prefix or t.startswith(prefix + " ") or t.startswith(prefix + "?"):
            return True
    # Very short + only filler words (e.g. "ok thanks")
    if len(t) <= 40 and re.fullmatch(
        r"(?:ok|okay|yes|yeah|thanks?|thx|sure|cool|nice|great)[\s,!.]+"
        r"(?:ok|okay|yes|yeah|thanks?|thx|sure|cool|nice|great|please|pls)[\s,!.]*",
        t,
    ):
        return True
    return False


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _count_tool_calls(messages: list[dict]) -> int:
    """Count tool invocations in assistant turns only (ignore developer instructions)."""
    n = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        n += len(re.findall(r'"type":\s*"tool_use"', str(m.get("content", ""))))
    return n


def format_example(tokenizer, messages: list[dict]) -> dict | None:
    """
    Return {"text": ...} with the full rendered conversation, or None if filtered out.

    Filters:
    - More than MAX_TOOL_CALLS tool_use blocks (keeps single-step patterns only)
    - More than MAX_MESSAGES turns (excludes long multi-turn extracts)
    - Exceeds MAX_TOKENS after rendering (measured with tiktoken o200k_base)

    mlx-lm re-applies the chat template when using {"prompt", "completion"} format,
    which causes the harmony template to reject the already-rendered <|channel|> tokens.
    Using {"text": ...} bypasses template processing entirely and trains on the raw sequence.
    """
    assert messages[-1]["role"] == "assistant", "Last message must be from assistant"

    if _count_tool_calls(messages) > MAX_TOOL_CALLS:
        return None

    if len(messages) > MAX_MESSAGES:
        return None

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    if len(ENCODING.encode(text)) > MAX_TOKENS:
        return None

    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument(
        "--input-dir",
        default=str(Path(__file__).parent.parent / "custom-data"),
    )
    parser.add_argument("--output-dir", default=str(Path(__file__).parent.parent / "data"))
    parser.add_argument("--valid-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model} …")
    tokenizer = load_tokenizer(args.model)

    input_files = sorted(Path(args.input_dir).glob("*.jsonl"))
    print(f"Input files: {[f.name for f in input_files]}")

    examples = []
    seen_first_msg: set[str] = set()
    skipped_followup = 0
    for input_file in input_files:
        with open(input_file) as f:
            for i, line in enumerate(f):
                raw = json.loads(line)

                # Pre-rendered examples ({"text": "..."}) — pass through with token check only
                if "text" in raw and "messages" not in raw:
                    text = raw["text"]
                    if len(ENCODING.encode(text)) <= MAX_TOKENS:
                        examples.append({"text": text})
                    continue

                msgs = raw.get("messages", [])
                first_user_full = next(
                    (str(m["content"]) for m in msgs if m["role"] == "user"), ""
                )
                if is_followup_first_user(first_user_full):
                    skipped_followup += 1
                    continue
                first_user = first_user_full[:120]
                if first_user in seen_first_msg:
                    continue
                seen_first_msg.add(first_user)
                try:
                    ex = format_example(tokenizer, msgs)
                    if ex is not None:
                        examples.append(ex)
                except Exception as e:
                    print(f"  Skipping {input_file.name}:{i}: {e}")

    print(
        f"Formatted {len(examples)} examples (after cross-file dedup, "
        f"skipped {skipped_followup} follow-up first-user rows)."
    )

    random.seed(args.seed)
    random.shuffle(examples)
    split = max(1, int(len(examples) * args.valid_split))
    valid = examples[:split]
    train = examples[split:]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, subset in [("train", train), ("valid", valid)]:
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in subset:
                f.write(json.dumps(ex) + "\n")
        print(f"  {path}: {len(subset)} examples")

    print("\nToken stats (tiktoken o200k_base):")
    all_examples = train + valid
    token_counts = sorted(len(ENCODING.encode(ex["text"])) for ex in all_examples)
    n = len(token_counts)
    print(f"  min={token_counts[0]}  median={token_counts[n//2]}  mean={sum(token_counts)//n}  max={token_counts[-1]}")

    print("\nSample text (first train example, last 300 chars):")
    print(train[0]["text"][-300:])


if __name__ == "__main__":
    main()
