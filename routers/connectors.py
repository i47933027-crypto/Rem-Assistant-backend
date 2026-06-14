from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel
from typing import Optional
from database import get_db
from search_engine import (
    web_search, news_search,
    github_search_repos, github_search_code, github_user,
    call_connector,
)
from config import settings
import json

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# We store connectors in the DB via a simple JSON column
# The Connector model is defined in database.py (added below)
# ─────────────────────────────────────────────────────────────

class ConnectorCreate(BaseModel):
    name: str
    description: str = ""
    icon: str = "⚙"
    base_url: str
    endpoint: str = "/"
    method: str = "GET"
    headers: dict = {}
    params_template: dict = {}
    body_template: dict = {}
    api_key: Optional[str] = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    enabled: bool = True

class ConnectorCall(BaseModel):
    query: Optional[str] = None
    params: dict = {}
    body: dict = {}


# ════════════════════════════════════════════════════════════════
# BUILT-IN: WEB SEARCH
# ════════════════════════════════════════════════════════════════

@router.get("/search/web")
async def search_web(q: str, limit: int = 6):
    """DuckDuckGo web search — free, no key needed."""
    result = await web_search(q, max_results=limit)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "Search failed"))
    return result

@router.get("/search/news")
async def search_news(q: str, limit: int = 5):
    """DuckDuckGo news search — free, no key needed."""
    result = await news_search(q, max_results=limit)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "News search failed"))
    return result


# ════════════════════════════════════════════════════════════════
# BUILT-IN: GITHUB SEARCH
# ════════════════════════════════════════════════════════════════

@router.get("/search/github")
async def search_github(q: str, type: str = "repos", limit: int = 5):
    """GitHub search — free (60 req/hr), add GITHUB_TOKEN for 5000/hr."""
    token = getattr(settings, "GITHUB_TOKEN", None)
    if type == "code":
        result = await github_search_code(q, limit, token)
    else:
        result = await github_search_repos(q, limit=limit, token=token)
    if not result["success"]:
        raise HTTPException(400, result.get("error", "GitHub search failed"))
    return result

@router.get("/search/github/user/{username}")
async def github_profile(username: str):
    """Get GitHub user profile."""
    token = getattr(settings, "GITHUB_TOKEN", None)
    result = await github_user(username)
    if not result["success"]:
        raise HTTPException(404, result.get("error", "User not found"))
    return result


# ════════════════════════════════════════════════════════════════
# CUSTOM CONNECTORS — CRUD
# ════════════════════════════════════════════════════════════════

@router.get("")
async def list_connectors(db: AsyncSession = Depends(get_db)):
    """List all saved custom connectors."""
    from database import Connector
    result = await db.execute(select(Connector).order_by(Connector.created_at.desc()))
    connectors = result.scalars().all()
    return [_serialize(c) for c in connectors]


@router.post("")
async def create_connector(body: ConnectorCreate, db: AsyncSession = Depends(get_db)):
    """Save a new custom connector."""
    from database import Connector
    c = Connector(
        name=body.name,
        description=body.description,
        icon=body.icon,
        base_url=body.base_url,
        endpoint=body.endpoint,
        method=body.method.upper(),
        headers=body.headers,
        params_template=body.params_template,
        body_template=body.body_template,
        api_key=body.api_key,
        api_key_header=body.api_key_header,
        api_key_prefix=body.api_key_prefix,
        enabled=body.enabled,
    )
    db.add(c)
    await db.commit()
    await db.refresh(c)
    return _serialize(c)


@router.put("/{connector_id}")
async def update_connector(connector_id: int, body: ConnectorCreate, db: AsyncSession = Depends(get_db)):
    from database import Connector
    result = await db.execute(select(Connector).where(Connector.id == connector_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(404, "Connector not found")
    for field, val in body.dict().items():
        setattr(c, field, val)
    if hasattr(c, "method"):
        c.method = c.method.upper()
    await db.commit()
    return _serialize(c)


@router.delete("/{connector_id}")
async def delete_connector(connector_id: int, db: AsyncSession = Depends(get_db)):
    from database import Connector
    await db.execute(delete(Connector).where(Connector.id == connector_id))
    await db.commit()
    return {"status": "deleted"}


@router.post("/{connector_id}/test")
async def test_connector(connector_id: int, db: AsyncSession = Depends(get_db)):
    """Test a connector with a simple request."""
    from database import Connector
    result = await db.execute(select(Connector).where(Connector.id == connector_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(404, "Connector not found")
    resp = await call_connector(
        base_url=c.base_url,
        endpoint=c.endpoint,
        method=c.method,
        headers=c.headers or {},
        params=c.params_template or {},
        api_key=c.api_key,
        api_key_header=c.api_key_header,
        api_key_prefix=c.api_key_prefix,
    )
    return {"connector": c.name, "test_result": resp}


@router.post("/{connector_id}/call")
async def call_connector_endpoint(connector_id: int, body: ConnectorCall, db: AsyncSession = Depends(get_db)):
    """Call a connector with optional query/params."""
    from database import Connector
    result = await db.execute(select(Connector).where(Connector.id == connector_id))
    c = result.scalar_one_or_none()
    if not c:
        raise HTTPException(404, "Connector not found")
    if not c.enabled:
        raise HTTPException(400, "Connector is disabled")

    # Merge template params with request params
    params = dict(c.params_template or {})
    params.update(body.params)
    if body.query:
        # Auto-inject query into common param names
        for key in ["q", "query", "search", "text", "input", "prompt"]:
            if key in params:
                params[key] = body.query
                break
        else:
            params["q"] = body.query

    merged_body = dict(c.body_template or {})
    merged_body.update(body.body)

    resp = await call_connector(
        base_url=c.base_url,
        endpoint=c.endpoint,
        method=c.method,
        headers=c.headers or {},
        params=params,
        body=merged_body if merged_body else None,
        api_key=c.api_key,
        api_key_header=c.api_key_header,
        api_key_prefix=c.api_key_prefix,
    )
    return resp


def _serialize(c) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "description": c.description,
        "icon": c.icon,
        "base_url": c.base_url,
        "endpoint": c.endpoint,
        "method": c.method,
        "headers": c.headers or {},
        "params_template": c.params_template or {},
        "body_template": c.body_template or {},
        "api_key_header": c.api_key_header,
        "api_key_prefix": c.api_key_prefix,
        "has_api_key": bool(c.api_key),
        "enabled": c.enabled,
        "created_at": str(c.created_at),
    }
