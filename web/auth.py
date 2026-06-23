"""FastAPI dependency for X-API-Key authentication.

Usage:
    from web.auth import verify_api_key

    @router.post("/protected", dependencies=[Depends(verify_api_key)])
    async def protected_endpoint():
        ...
"""

from db.engine import get_session
from db.models import ApiKey
from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session


async def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_session),
) -> ApiKey:
    """FastAPI dependency that validates the X-API-Key header.

    Returns the ApiKey row on success; raises 401 on failure.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-API-Key header",
        )

    key_hash = ApiKey.hash_key(x_api_key)
    result = db.execute(select(ApiKey).where(ApiKey.key_hash == key_hash))
    api_key: ApiKey | None = result.scalars().first()

    if api_key is None or not api_key.active:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key",
        )

    return api_key
