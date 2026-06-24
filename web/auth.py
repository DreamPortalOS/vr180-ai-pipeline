"""API Key authentication for VR180 Studio.

Uses ``X-API-Key`` header for write-operation authentication.
Keys are stored as bcrypt hashes; the plaintext is shown only once at creation.
"""

import logging

from db.engine import get_db
from db.models import APIKey
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from passlib.context import CryptContext
from sqlalchemy.orm import Session

log = logging.getLogger("vr180-auth")

# ── Passlib bcrypt context ─────────────────────────────────────────────────────
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_key(plain: str) -> str:
    """Return bcrypt hash of *plain*."""
    return _pwd.hash(plain)


def verify_key(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*."""
    return _pwd.verify(plain, hashed)


# ── FastAPI security scheme ────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    x_api_key: str | None = Depends(_api_key_header),
    db: Session = Depends(get_db),
) -> str | None:
    """FastAPI dependency: validate ``X-API-Key`` header against the database.

    Returns the API key plaintext on success (caller may use it for logging).
    Raises ``401`` if the header is missing or the key is invalid/inactive.

    Write-operation endpoints inject this dependency via ``Depends(verify_api_key)``.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Look up by plaintext key first (legacy), then by hash comparison
    record = db.query(APIKey).filter(APIKey.key == x_api_key).first()
    if record is None:
        # Search by hash: iterate active keys and compare
        for candidate in db.query(APIKey).filter(APIKey.is_active.is_(True)).all():
            if candidate.key_hash and verify_key(x_api_key, candidate.key_hash):
                record = candidate
                break

    if record is None or not record.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key
