#!/usr/bin/env python3
"""
Seed both vector backends by posting .txt files in this directory to /ingest.

Usage:
    python3 seeds/seed.py <api_url> [user_id]
"""

import json
import pathlib
import sys
import urllib.error
import urllib.request

if len(sys.argv) < 2:
    print("Usage: seed.py <api_url> [user_id]")
    sys.exit(1)

API_URL = sys.argv[1].rstrip("/")
USER_ID = sys.argv[2] if len(sys.argv) > 2 else "test"

seed_files = sorted(pathlib.Path(__file__).parent.glob("*.txt"))
if not seed_files:
    print("No .txt files found in seeds/")
    sys.exit(1)

print(f"Seeding {len(seed_files)} document(s) as user '{USER_ID}'...")

for path in seed_files:
    payload = json.dumps(
        {
            "user_id": USER_ID,
            "document_id": path.stem,
            "text": path.read_text(),
        }
    ).encode()

    req = urllib.request.Request(
        f"{API_URL}/ingest",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
        print(
            f"{path.stem}: accepted (document_id={resp.get('document_id', path.stem)}, status={resp.get('status')})"
        )
    except urllib.error.HTTPError as e:
        print(f"{path.stem}: ERROR {e.code} — {e.read().decode()}")

print("Done.")
