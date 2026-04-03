#!/usr/bin/env python3
"""Simple interactive chat CLI for the local model server."""
import argparse
import json
import urllib.request
import urllib.error

def chat(base_url: str, model: str) -> None:
    print(f"Chatting with {model} at {base_url}")
    print("Type your message and press Enter. Ctrl-C or 'exit' to quit.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input or user_input.lower() == "exit":
            break

        history.append({"role": "user", "content": user_input})

        payload = json.dumps({
            "model": model,
            "max_tokens": 2000,
            "temperature": 0,
            "messages": history,
        }).encode()

        req = urllib.request.Request(
            f"{base_url}/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json", "x-api-key": "local"},
        )

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            reply = data["content"][0]["text"]
            print(f"\nModel: {reply}\n")
            history.append({"role": "assistant", "content": reply})
        except urllib.error.URLError as e:
            print(f"Error: {e}\n")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Unexpected response: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive chat with local model")
    parser.add_argument("--url", default="http://localhost:8082", help="Base URL of the server")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    args = parser.parse_args()
    chat(args.url, args.model)
