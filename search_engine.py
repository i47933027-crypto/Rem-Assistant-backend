"""
REM OS — SEARCH ENGINE
Built-in: DuckDuckGo (free), GitHub (free)
Custom: any REST API the user adds
"""
import httpx
import asyncio
from duckduckgo_search import DDGS
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# WEB SEARCH — DuckDuckGo (completely free, no key needed)
# ═══════════════════════════════════════════════════════════════

async def web_search(query: str, max_results: int = 6) -> dict:
    loop = asyncio.get_event_loop()
    def _search():
        with DDGS() as ddg:
            return list(ddg.text(query, max_results=max_results))
    try:
        results = await loop.run_in_executor(None, _search)
        return {
            "success": True,
            "query": query,
            "source": "duckduckgo",
            "results": [{"title": r["title"], "snippet": r["body"], "url": r["href"]} for r in results],
            "count": len(results),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "results": []}


async def news_search(query: str, max_results: int = 5) -> dict:
    loop = asyncio.get_event_loop()
    def _search():
        with DDGS() as ddg:
            return list(ddg.news(query, max_results=max_results))
    try:
        results = await loop.run_in_executor(None, _search)
        return {
            "success": True,
            "query": query,
            "source": "duckduckgo_news",
            "results": [{"title": r["title"], "snippet": r.get("body",""), "url": r.get("url",""), "date": r.get("date","")} for r in results],
            "count": len(results),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "results": []}


# ═══════════════════════════════════════════════════════════════
# GITHUB SEARCH — free, no key needed (60 req/hr unauthenticated)
# ═══════════════════════════════════════════════════════════════

async def github_search_repos(query: str, sort: str = "stars", limit: int = 5, token: Optional[str] = None) -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=15.0) as c:
        try:
            r = await c.get(
                "https://api.github.com/search/repositories",
                headers=headers,
                params={"q": query, "sort": sort, "per_page": limit},
            )
            d = r.json()
            if "items" not in d:
                return {"success": False, "error": d.get("message", "GitHub error"), "results": []}
            items = d["items"]
            return {
                "success": True,
                "query": query,
                "source": "github_repos",
                "total": d.get("total_count", len(items)),
                "results": [{
                    "name": i["full_name"],
                    "description": i.get("description") or "No description",
                    "url": i["html_url"],
                    "stars": i["stargazers_count"],
                    "language": i.get("language") or "Unknown",
                    "topics": i.get("topics", [])[:4],
                    "updated": i.get("updated_at", "")[:10],
                } for i in items],
            }
        except Exception as e:
            return {"success": False, "error": str(e), "results": []}


async def github_search_code(query: str, limit: int = 5, token: Optional[str] = None) -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=15.0) as c:
        try:
            r = await c.get(
                "https://api.github.com/search/code",
                headers=headers,
                params={"q": query, "per_page": limit},
            )
            d = r.json()
            if "items" not in d:
                return {"success": False, "error": d.get("message", "GitHub error"), "results": []}
            return {
                "success": True,
                "query": query,
                "source": "github_code",
                "results": [{
                    "name": i["name"],
                    "path": i["path"],
                    "repo": i["repository"]["full_name"],
                    "url": i["html_url"],
                } for i in d["items"]],
            }
        except Exception as e:
            return {"success": False, "error": str(e), "results": []}


async def github_user(username: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as c:
        try:
            r = await c.get(f"https://api.github.com/users/{username}", headers={"Accept": "application/vnd.github.v3+json"})
            d = r.json()
            if "login" not in d:
                return {"success": False, "error": d.get("message", "User not found")}
            return {"success": True, "source": "github_user", "user": {"name": d.get("name"), "bio": d.get("bio"), "repos": d.get("public_repos"), "followers": d.get("followers"), "url": d.get("html_url")}}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL CONNECTOR — call any REST API
# ═══════════════════════════════════════════════════════════════

async def call_connector(
    base_url: str,
    endpoint: str,
    method: str = "GET",
    headers: dict = None,
    params: dict = None,
    body: dict = None,
    api_key: Optional[str] = None,
    api_key_header: str = "Authorization",
    api_key_prefix: str = "Bearer",
) -> dict:
    """Call any REST API connector."""
    all_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        all_headers.update(headers)
    if api_key:
        all_headers[api_key_header] = f"{api_key_prefix} {api_key}".strip()

    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    async with httpx.AsyncClient(timeout=30.0) as c:
        try:
            if method.upper() == "GET":
                r = await c.get(url, headers=all_headers, params=params)
            elif method.upper() == "POST":
                r = await c.post(url, headers=all_headers, params=params, json=body)
            elif method.upper() == "PUT":
                r = await c.put(url, headers=all_headers, json=body)
            elif method.upper() == "DELETE":
                r = await c.delete(url, headers=all_headers)
            else:
                return {"success": False, "error": f"Unsupported method: {method}"}

            try:
                data = r.json()
            except Exception:
                data = r.text

            return {
                "success": r.status_code < 400,
                "status_code": r.status_code,
                "data": data,
                "url": url,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}


def build_search_context(web_results: dict, github_results: Optional[dict] = None) -> str:
    """Build clean context string from search results for LLM."""
    parts = []
    if web_results.get("success") and web_results.get("results"):
        parts.append(f"[WEB SEARCH: {web_results['query']}]")
        for i, r in enumerate(web_results["results"][:5], 1):
            parts.append(f"{i}. {r['title']}\n   {r['snippet'][:200]}\n   Source: {r['url']}")
    if github_results and github_results.get("success") and github_results.get("results"):
        parts.append(f"\n[GITHUB: {github_results['query']}]")
        for r in github_results["results"][:4]:
            parts.append(f"• {r['name']} ⭐{r.get('stars',0):,} — {r.get('description','')[:100]}\n  {r['url']}")
    return "\n".join(parts)
