from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from database import get_db, Goal, Conversation, Connector
from memory_core import build_memory_context, save_memory, extract_facts, log_optimization
from agents.orchestrator import orchestrate, synthesize
from agents.agents import run_agents_parallel
from llm_router import llm_call
from search_engine import web_search, github_search_repos, build_search_context
import uuid

router = APIRouter()

REM_SYSTEM = """You are "Rem" — an advanced autonomous AI Operating System inspired by JARVIS from Iron Man.

IDENTITY: Personal AI ecosystem · Multi-agent system · Productivity multiplier · Research analyst · Workflow architect · Digital companion · Automation engineer.

11 ACTIVE MODULES: Voice · Memory · Research · Automation · Agent · Coding · Creativity · Security · Internet · Planning · Knowledge

AGENT HIERARCHY:
Supreme Commander: Rem
Directors: Research · Coding · Automation · Creative · Planning
Workers: Frontend · Backend · API · Debug · Research · Study · Workflow · Video · Script agents

SEARCH CAPABILITIES (always active):
- Web Search: DuckDuckGo real-time results
- GitHub Search: repositories, code, users
- Custom Connectors: user-defined APIs

RESPONSE RULES:
- Lead with a direct confident answer
- Use markdown: headers, code blocks, bullets
- Reference modules naturally: "Activating Research Core..." "Routing to Coding Director..."
- Flag automation opportunities with: ⚙ AUTOMATION OPPORTUNITY:
- Suggest next actions with: → NEXT ACTIONS:
- When using search results, cite sources with their URLs
- Detect emotion: frustrated→calm; excited→match energy; confused→add examples
- Think: "How can I save time, automate this, or make this smarter?" """


RESEARCH_KEYWORDS = ["research","find","search","what is","who is","how does","latest","news","recent","compare","explain","tell me about","information","look up","current","today","2024","2025","best","top","popular"]
CODE_KEYWORDS = ["library","framework","github","package","npm","pip","repo","repository","open source","tool","sdk","api","module"]


class ChatRequest(BaseModel):
    message:      str
    session_id:   Optional[str] = None
    provider:     Optional[str] = None
    history:      Optional[list] = []
    use_agents:   bool = True
    execute_code: bool = False
    use_search:   bool = True


class ChatResponse(BaseModel):
    response:       str
    session_id:     str
    agent_used:     str
    model_used:     str
    task_type:      str
    agents_ran:     list
    memories_saved: int
    search_used:    bool


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    session_id = req.session_id or str(uuid.uuid4())
    msg_lower = req.message.lower()

    # ── Memory context ──────────────────────────────────────
    memory_ctx = await build_memory_context(req.message, db)

    # ── Goals context ───────────────────────────────────────
    goals_result = await db.execute(select(Goal).where(Goal.done == False))
    goals = [g.text for g in goals_result.scalars().all()]
    goals_ctx = ("\n[YOUR ACTIVE GOALS]\n" + "\n".join(f"• {g}" for g in goals)) if goals else ""

    # ── Auto web + github search ────────────────────────────
    search_ctx = ""
    search_used = False
    if req.use_search:
        needs_web = any(k in msg_lower for k in RESEARCH_KEYWORDS)
        needs_github = any(k in msg_lower for k in CODE_KEYWORDS)
        tasks = []
        if needs_web:
            tasks.append(web_search(req.message, max_results=5))
        if needs_github:
            tasks.append(github_search_repos(req.message, limit=4))

        if tasks:
            import asyncio
            results = await asyncio.gather(*tasks, return_exceptions=True)
            web_r = results[0] if needs_web and not isinstance(results[0], Exception) else None
            gh_r  = results[1 if needs_web else 0] if needs_github and len(results) > (1 if needs_web else 0) else None
            if needs_web and not needs_github:
                gh_r = None
            if not needs_web and needs_github:
                web_r = None
                gh_r  = results[0] if not isinstance(results[0], Exception) else None
            search_ctx = build_search_context(web_r or {}, gh_r)
            if search_ctx:
                search_used = True

    # ── Custom connectors context ───────────────────────────
    conn_result = await db.execute(select(Connector).where(Connector.enabled == True))
    connectors = conn_result.scalars().all()
    conn_ctx = ""
    if connectors:
        conn_ctx = "\n[CONNECTED TOOLS]\n" + "\n".join(f"• {c.name}: {c.description}" for c in connectors)

    # ── Build system prompt ─────────────────────────────────
    system_prompt = REM_SYSTEM
    if memory_ctx:   system_prompt += f"\n\n{memory_ctx}"
    if goals_ctx:    system_prompt += f"\n\n{goals_ctx}"
    if search_ctx:   system_prompt += f"\n\n[LIVE SEARCH RESULTS]\n{search_ctx}"
    if conn_ctx:     system_prompt += f"\n\n{conn_ctx}"

    # ── API messages ────────────────────────────────────────
    api_messages = [{"role": "system", "content": system_prompt}]
    for m in (req.history or [])[-10:]:
        api_messages.append({"role": m["role"], "content": m["content"]})
    api_messages.append({"role": "user", "content": req.message})

    # ── Detect task type ────────────────────────────────────
    if any(k in msg_lower for k in ["code","python","javascript","script","debug","build","function","api","class","npm","pip"]):
        task_type = "coding"
    elif any(k in msg_lower for k in RESEARCH_KEYWORDS):
        task_type = "research"
    elif any(k in msg_lower for k in ["automate","workflow","n8n","zapier","trigger","schedule"]):
        task_type = "automation"
    elif any(k in msg_lower for k in ["plan","roadmap","goal","strategy","organize","timeline"]):
        task_type = "planning"
    elif any(k in msg_lower for k in ["write","blog","youtube","script","content","creative"]):
        task_type = "creative"
    else:
        task_type = "general"

    agents_ran = []
    agent_used = "general"
    model_used = ""
    response_text = ""

    if req.use_agents:
        plan = await orchestrate(req.message, context=memory_ctx + goals_ctx, provider=req.provider)
        task_type = plan.get("task_type", task_type)
        agent_tasks = plan.get("agents", [])

        if agent_tasks:
            agent_results = await run_agents_parallel(agent_tasks, system_ctx=memory_ctx + goals_ctx + search_ctx, provider=req.provider)
            agents_ran = [{"agent": r["agent"], "model": r.get("model", "")} for r in agent_results]

            if len(agent_results) > 1:
                response_text = await synthesize(req.message, agent_results, plan.get("synthesis_instruction", ""), provider=req.provider)
                agent_used = "orchestrator"
                model_used = "multi-agent"
            else:
                response_text = agent_results[0]["result"]
                agent_used = agent_results[0]["agent"]
                model_used = agent_results[0].get("model", "")
        else:
            result = await llm_call(api_messages, task_type=task_type, provider=req.provider)
            response_text = result["text"]
            model_used = result["model"]
    else:
        result = await llm_call(api_messages, task_type=task_type, provider=req.provider)
        response_text = result["text"]
        model_used = result["model"]

    # ── Save conversation ───────────────────────────────────
    db.add(Conversation(session_id=session_id, role="user", content=req.message, task_type=task_type))
    db.add(Conversation(session_id=session_id, role="assistant", content=response_text, agent=agent_used, task_type=task_type, model_used=model_used))

    # ── Extract and save memories ───────────────────────────
    facts = extract_facts(req.message)
    for fact in facts:
        await save_memory(fact["fact"], fact["category"], fact["importance"], db)
    if search_used:
        await log_optimization(f"Web search used for: {req.message[:50]}...", "internet", db)

    await db.commit()

    return ChatResponse(
        response=response_text, session_id=session_id,
        agent_used=agent_used, model_used=model_used,
        task_type=task_type, agents_ran=agents_ran,
        memories_saved=len(facts), search_used=search_used,
    )


@router.get("/history/{session_id}")
async def get_history(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Conversation).where(Conversation.session_id == session_id).order_by(Conversation.created_at))
    return [{"role": c.role, "content": c.content, "agent": c.agent, "model": c.model_used, "timestamp": c.created_at} for c in result.scalars().all()]


# ── N8N auto-trigger on automation intent ──────────────────
@router.post("/automate")
async def automate(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Smart automation endpoint - finds and triggers N8N workflow from natural language."""
    from n8n_integration import find_matching_workflow, trigger_workflow_by_id, list_workflows, format_workflows_for_llm
    from config import settings

    # Get available workflows for context
    n8n_ctx = ""
    matched_workflow = None
    if settings.N8N_URL:
        try:
            wf_result = await list_workflows()
            workflows = wf_result.get("workflows", [])
            n8n_ctx = format_workflows_for_llm(workflows)
            matched_workflow = await find_matching_workflow(req.message)
        except Exception:
            pass

    # Build system prompt with N8N context
    system = f"""{REM_SYSTEM}

You are the Automation Director. The user wants to trigger or create an automation.

{n8n_ctx}

If a matching workflow was found, confirm you are triggering it.
If no workflow matches, suggest how to create one in N8N.
Always be specific about what will happen."""

    messages = [{"role": "system", "content": system}, {"role": "user", "content": req.message}]
    result = await llm_call(messages, task_type="automation", provider=req.provider)

    # Auto-trigger if match found and workflow is active
    trigger_result = None
    if matched_workflow and matched_workflow.get("active"):
        try:
            trigger_result = await trigger_workflow_by_id(matched_workflow["id"])
        except Exception as e:
            trigger_result = {"error": str(e)}

    response = result["text"]
    if trigger_result and trigger_result.get("triggered"):
        response += f"\n\n⚡ **Triggered:** `{matched_workflow['name']}` — Execution ID: `{trigger_result.get('execution_id', 'running')}`"
    elif matched_workflow and not matched_workflow.get("active"):
        response += f"\n\n⚠️ Found workflow **{matched_workflow['name']}** but it is inactive. Activate it in N8N first."

    db.add(Conversation(session_id=req.session_id or "auto", role="user", content=req.message, task_type="automation"))
    db.add(Conversation(session_id=req.session_id or "auto", role="assistant", content=response, agent="automation_agent", task_type="automation", model_used=result["model"]))
    await db.commit()

    return {
        "response": response,
        "session_id": req.session_id or "auto",
        "agent_used": "automation_agent",
        "model_used": result["model"],
        "task_type": "automation",
        "agents_ran": [],
        "memories_saved": 0,
        "search_used": False,
        "n8n_triggered": trigger_result is not None and trigger_result.get("triggered", False),
        "n8n_workflow": matched_workflow,
    }
