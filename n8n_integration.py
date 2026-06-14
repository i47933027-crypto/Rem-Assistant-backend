"""
REM OS — N8N INTEGRATION
Real N8N workflow triggering, management, and execution tracking
"""
import httpx
from typing import Optional
from config import settings


def n8n_headers():
    return {
        "X-N8N-API-KEY": settings.N8N_API_KEY or "",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def n8n_base():
    return (settings.N8N_URL or "").rstrip("/")


# ══════════════════════════════════════════════════════════════
# CONNECTION
# ══════════════════════════════════════════════════════════════

async def test_connection() -> dict:
    if not settings.N8N_URL:
        return {"connected": False, "error": "N8N_URL not configured"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{n8n_base()}/api/v1/workflows", headers=n8n_headers())
            if r.status_code == 200:
                data = r.json()
                count = len(data.get("data", []))
                return {"connected": True, "url": settings.N8N_URL, "workflow_count": count}
            return {"connected": False, "error": f"Status {r.status_code}: {r.text[:100]}"}
    except Exception as e:
        return {"connected": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# WORKFLOWS
# ══════════════════════════════════════════════════════════════

async def list_workflows() -> dict:
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get(f"{n8n_base()}/api/v1/workflows", headers=n8n_headers())
        if r.status_code != 200:
            raise Exception(f"N8N error {r.status_code}: {r.text[:200]}")
        data = r.json()
        workflows = data.get("data", [])
        return {
            "count": len(workflows),
            "workflows": [{
                "id":          w["id"],
                "name":        w["name"],
                "active":      w.get("active", False),
                "created_at":  w.get("createdAt", "")[:10],
                "updated_at":  w.get("updatedAt", "")[:10],
                "tags":        [t["name"] for t in w.get("tags", [])],
                "node_count":  len(w.get("nodes", [])),
            } for w in workflows]
        }


async def get_workflow(workflow_id: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{n8n_base()}/api/v1/workflows/{workflow_id}", headers=n8n_headers())
        if r.status_code != 200:
            raise Exception(f"Workflow not found: {r.text[:100]}")
        return r.json()


async def activate_workflow(workflow_id: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.post(f"{n8n_base()}/api/v1/workflows/{workflow_id}/activate", headers=n8n_headers())
        if r.status_code not in (200, 201):
            raise Exception(f"Failed to activate: {r.text[:100]}")
        return {"status": "activated", "id": workflow_id}


async def deactivate_workflow(workflow_id: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.post(f"{n8n_base()}/api/v1/workflows/{workflow_id}/deactivate", headers=n8n_headers())
        if r.status_code not in (200, 201):
            raise Exception(f"Failed to deactivate: {r.text[:100]}")
        return {"status": "deactivated", "id": workflow_id}


# ══════════════════════════════════════════════════════════════
# TRIGGER WORKFLOWS
# ══════════════════════════════════════════════════════════════

async def trigger_webhook(webhook_path: str, payload: dict = None) -> dict:
    """Trigger an N8N webhook workflow."""
    url = f"{n8n_base()}/webhook/{webhook_path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(url, json=payload or {}, headers={"Content-Type": "application/json"})
        try:
            response_data = r.json()
        except Exception:
            response_data = {"raw": r.text[:500]}
        return {
            "triggered":   True,
            "webhook_url": url,
            "status_code": r.status_code,
            "response":    response_data,
            "success":     r.status_code < 400,
        }


async def trigger_workflow_by_id(workflow_id: str, data: dict = None) -> dict:
    """Execute a workflow by ID using N8N API."""
    async with httpx.AsyncClient(timeout=30.0) as c:
        payload = {"workflowData": data or {}}
        r = await c.post(
            f"{n8n_base()}/api/v1/executions",
            headers=n8n_headers(),
            json={"workflowId": workflow_id, **payload}
        )
        if r.status_code in (200, 201):
            d = r.json()
            return {"triggered": True, "execution_id": d.get("id"), "status": d.get("status", "running")}
        raise Exception(f"Trigger failed {r.status_code}: {r.text[:200]}")


# ══════════════════════════════════════════════════════════════
# EXECUTIONS
# ══════════════════════════════════════════════════════════════

async def list_executions(limit: int = 10, workflow_id: Optional[str] = None) -> dict:
    params = {"limit": limit}
    if workflow_id:
        params["workflowId"] = workflow_id
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{n8n_base()}/api/v1/executions", headers=n8n_headers(), params=params)
        if r.status_code != 200:
            raise Exception(f"N8N error: {r.text[:100]}")
        data = r.json()
        executions = data.get("data", [])
        return {
            "count": len(executions),
            "executions": [{
                "id":          e["id"],
                "workflow_id": e.get("workflowId", ""),
                "status":      e.get("status", "unknown"),
                "started_at":  e.get("startedAt", "")[:19] if e.get("startedAt") else "",
                "stopped_at":  e.get("stoppedAt", "")[:19] if e.get("stoppedAt") else "",
                "finished":    e.get("finished", False),
                "mode":        e.get("mode", ""),
            } for e in executions]
        }


async def get_execution(execution_id: str) -> dict:
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(f"{n8n_base()}/api/v1/executions/{execution_id}", headers=n8n_headers())
        if r.status_code != 200:
            raise Exception(f"Execution not found: {r.text[:100]}")
        d = r.json()
        return {
            "id":         d.get("id"),
            "status":     d.get("status", "unknown"),
            "finished":   d.get("finished", False),
            "started_at": d.get("startedAt", "")[:19],
            "data":       d.get("data", {}),
        }


# ══════════════════════════════════════════════════════════════
# SMART INTENT MATCHING
# ══════════════════════════════════════════════════════════════

async def find_matching_workflow(intent: str) -> Optional[dict]:
    """Find best matching workflow for a user's intent."""
    try:
        result = await list_workflows()
        workflows = result.get("workflows", [])
        if not workflows:
            return None
        intent_lower = intent.lower()
        # Score each workflow by name/tag relevance
        scored = []
        for w in workflows:
            score = 0
            name_lower = w["name"].lower()
            words = intent_lower.split()
            for word in words:
                if len(word) > 3 and word in name_lower:
                    score += 2
            for tag in w.get("tags", []):
                if tag.lower() in intent_lower:
                    score += 3
            if w.get("active"):
                score += 1
            if score > 0:
                scored.append((score, w))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
        return None
    except Exception:
        return None


def format_workflows_for_llm(workflows: list) -> str:
    """Format workflow list for injection into LLM context."""
    if not workflows:
        return ""
    lines = ["[N8N WORKFLOWS AVAILABLE]"]
    for w in workflows[:10]:
        status = "✅ Active" if w.get("active") else "⏸ Inactive"
        lines.append(f"• [{w['id']}] {w['name']} — {status} — {w.get('node_count', 0)} nodes")
    return "\n".join(lines)
