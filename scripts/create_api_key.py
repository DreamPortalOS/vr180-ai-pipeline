#!/usr/bin/env python3
"""CLI to generate a new API key and store it in the database.

Usage:
    python scripts/create_api_key.py --name "my-integration-key" [--user-id <id>]

The plaintext key is printed **once** and cannot be retrieved later.
"""

import argparse
import secrets

from db.engine import SessionLocal, init_db
from db.models import APIKey, User
from web.auth import hash_key


def _get_or_create_user(db, user_id: str):
    """Return an existing user or create one with *user_id*."""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        user = User(id=user_id)
        db.add(user)
        db.flush()
    return user


def main():
    parser = argparse.ArgumentParser(description="Generate a new API key")
    parser.add_argument(
        "--name",
        default="default",
        help="Human-friendly label for the key (default: 'default')",
    )
    parser.add_argument(
        "--user-id",
        default="default-user",
        help="Owner user ID; created if it doesn't exist (default: 'default-user')",
    )
    args = parser.parse_args()

    init_db()
    db = SessionLocal()

    try:
        # Ensure the owning user exists
        user = _get_or_create_user(db, args.user_id)

        # Generate a cryptographically random key
        plain_key = "vr180_" + secrets.token_hex(24)
        key_hash = hash_key(plain_key)

        api_key = APIKey(
            key_hash=key_hash,
            name=args.name,
            user_id=user.id,
            is_active=True,
        )
        db.add(api_key)
        db.commit()

        print("✅ API key created successfully!")
        print(f"   Name:        {api_key.name}")
        print(f"   User ID:     {api_key.user_id}")
        print(f"   Key ID:      {api_key.id}")
        print()
        print("   ╔══════════════════════════════════════════════════════╗")
        print("   ║  Plaintext key (shown ONLY once — SAVE IT NOW!):   ║")
        print("   ║                                                    ║")
        print(f"   ║  {plain_key:<50}║")
        print("   ╚══════════════════════════════════════════════════════╝")
        print()
        print("Use this key in the X-API-Key header for write operations.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
