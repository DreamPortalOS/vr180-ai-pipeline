#!/usr/bin/env python3
"""CLI tool to create an API key and store its hash in the database.

Usage:
    python scripts/create_api_key.py [--name my-key]

The raw key is printed ONCE.  Store it securely — it cannot be retrieved later.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.engine import SessionLocal, init_db
from db.models import ApiKey


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new API key")
    parser.add_argument("--name", default="default", help="Human-friendly key name")
    args = parser.parse_args()

    init_db()

    api_key_obj, raw_key = ApiKey.generate_key(name=args.name)
    with SessionLocal() as session:
        session.add(api_key_obj)
        session.commit()

    print("─" * 60)
    print(f"  API Key created  (name: {args.name})")
    print(f"  Key:  {raw_key}")
    print("─" * 60)
    print("⚠️  Store this key securely — it will NOT be shown again.")
    print()


if __name__ == "__main__":
    main()
