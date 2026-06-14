from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from n8n_integration import (
    test_connection, list_workflows, get_workflow,
    activate_workflow, deactivate_workflow,
    trigger_webhook, trigger_workflow_by_id,
    list_executions, get_execution, find_matching_workflow,
)
from config import settings

router = APIRouter()


# ══════════════════════════ STATUS ══════════════════════════

@router.get("/status")
async def n8n_status():
    """Check N8N connection status."""
    if not settings.N8N_URL:
        return {
            "connected": False,
            "configured": False,
            "message": "Add N8N_URL and N8N_API_KEY to your environment variables",
        }
    result = await test_connection()
    return {**result, "configured": True, "n8n_url": settings.N8N_URL}


# ══════════════════════════ WORKFLOWS ══════════════════════════

@router.get("/workflows")
async def get_workflows():
    """List all N8N workflows."""
    if not settings.N8N_URL:
        raise HTTPException(400, "N8N not configured. Add N8N_URL and N8N_API_KEY.")
    try:
        return await list_workflows()
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/workflows/{workflow_id}")
async def get_one_workflow(workflow_id: str):
    try:
        return await get_workflow(workflow_id)
    except Exception as e:
        raise HTTPException(404, str(e))


@router.post("/workflows/{workflow_id}/activate")
async def activate(workflow_id: str):
    try:
        return await activate_workflow(workflow_id)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/workflows/{workflow_id}/deactivate")
async def deactivate(workflow_id: str):
    try:
        return await deactivate_workflow(workflow_id)
    except Exception as e:
        raise HTTPException(400, str(e))


# ══════════════════════════ TRIGGER ══════════════════════════

class WebhookTrigger(BaseModel):
    payload: dict = {}

class WorkflowTrigger(BaseModel):
    data: dict = {}

class IntentTrigger(BaseModel):
    intent: str
    payload: dict = {}


@router.post("/trigger/webhook/{webhook_path:path}")
async def trigger_by_webhook(webhook_path: str, body: WebhookTrigger):
    """Trigger any N8N workflow via its webhook URL path."""
    if not settings.N8N_URL:
        raise HTTPException(400, "N8N not configured.")
    try:
        return await trigger_webhook(webhook_path, body.payload)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/trigger/workflow/{workflow_id}")
async def trigger_by_id(workflow_id: str, body: WorkflowTrigger):
    """Trigger an N8N workflow by its ID."""
    if not settings.N8N_URL:
        raise HTTPException(400, "N8N not configured.")
    try:
        return await trigger_workflow_by_id(workflow_id, body.data)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/trigger/intent")
async def trigger_by_intent(body: IntentTrigger):
    """Find and trigger the best matching workflow for a given intent."""
    if not settings.N8N_URL:
        raise HTTPException(400, "N8N not configured.")
    try:
        workflow = await find_matching_workflow(body.intent)
        if not workflow:
            return {"found": False, "message": "No matching workflow found for this intent"}
        if not workflow.get("active"):
            return {"found": True, "workflow": workflow, "triggered": False, "message": f"Found '{workflow['name']}' but it is inactive. Activate it first."}
        result = await trigger_workflow_by_id(workflow["id"], body.payload)
        return {"found": True, "workflow": workflow, "triggered": True, "result": result}
    except Exception as e:
        raise HTTPException(400, str(e))


# ══════════════════════════ EXECUTIONS ══════════════════════════

@router.get("/executions")
async def get_executions(limit: int = 10, workflow_id: Optional[str] = None):
    """List recent N8N executions."""
    if not settings.N8N_URL:
        raise HTTPException(400, "N8N not configured.")
    try:
        return await list_executions(limit, workflow_id)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/executions/{execution_id}")
async def get_one_execution(execution_id: str):
    try:
        return await get_execution(execution_id)
    except Exception as e:
        raise HTTPException(404, str(e))
