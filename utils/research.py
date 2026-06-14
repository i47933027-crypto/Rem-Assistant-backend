import httpx
import asyncio
from duckduckgo_search import DDGS


async def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a summary."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            s = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 2},
            )
            results = s.json().get("query", {}).get("search", [])
            if not results:
                return ""
            title = results[0]["title"]
            r = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}")
            data = r.json()
            extract = data.get("extract", "")[:800]
            related = [x["title"] for x in results[1:]]
            out = f"Wikipedia — {data.get('title', title)}\n{extract}"
            if related:
                out += f"\nRelated: {', '.join(related)}"
            return out
    except Exception as e:
        return ""


async def search_github(query: str) -> str:
    """Search GitHub repositories."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "per_page": 4},
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            items = r.json().get("items", [])
            if not items:
                return ""
            lines = [f"GitHub Top Repositories for '{query}':"]
            for item in items:
                lines.append(f"• {item['full_name']} ⭐{item['stargazers_count']:,} — {item.get('description', 'No description')[:80]}")
                lines.append(f"  URL: {item['html_url']}")
            return "\n".join(lines)
    except Exception:
        return ""


def search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo for current web results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        lines = [f"Web Search Results for '{query}':"]
        for r in results:
            lines.append(f"• {r['title']}")
            lines.append(f"  {r['body'][:200]}")
            lines.append(f"  Source: {r['href']}")
        return "\n".join(lines)
    except Exception as e:
        return ""


async def search_news(query: str) -> str:
    """Search DuckDuckGo news."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=4))
        if not results:
            return ""
        lines = [f"Latest News for '{query}':"]
        for r in results:
            lines.append(f"• [{r.get('date', '')}] {r['title']}")
            lines.append(f"  {r.get('body', '')[:150]}")
            lines.append(f"  Source: {r.get('url', '')}")
        return "\n".join(lines)
    except Exception:
        return ""


async def full_research(query: str, include_github: bool = False, include_news: bool = False) -> dict:
    """Run all research sources in parallel."""
    tasks = [search_wikipedia(query)]
    if include_github:
        tasks.append(search_github(query))
    if include_news:
        tasks.append(search_news(query))

    # DuckDuckGo is sync, run in executor
    loop = asyncio.get_event_loop()
    ddg_task = loop.run_in_executor(None, search_duckduckgo, query)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    ddg_result = await ddg_task

    output = {}
    output["wikipedia"] = results[0] if not isinstance(results[0], Exception) else ""
    if include_github:
        output["github"] = results[1] if not isinstance(results[1], Exception) else ""
    if include_news:
        output["news"] = results[-1] if not isinstance(results[-1], Exception) else ""
    output["web"] = ddg_result

    # Combine into context string
    parts = []
    if output.get("wikipedia"):
        parts.append(f"[WIKIPEDIA]\n{output['wikipedia']}")
    if output.get("web"):
        parts.append(f"[WEB SEARCH]\n{output['web']}")
    if output.get("github"):
        parts.append(f"[GITHUB]\n{output['github']}")
    if output.get("news"):
        parts.append(f"[NEWS]\n{output['news']}")

    output["combined"] = "\n\n".join(parts)
    return output
