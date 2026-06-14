import asyncio
from llm_router import llm_call
from utils.research import full_research
from utils.sandbox import execute_python
import re


# ═══════════════════════════════════════ RESEARCH AGENT ═══════════════════════════════════════
RESEARCH_SYS = """You are Rem's Research Director. You have access to live web data, Wikipedia, GitHub, and news sources.
Your job: analyze research data and produce clear, accurate, sourced insights.
Format: Use headers, bullet points, and cite sources. Be comprehensive but concise."""


async def research_agent(instruction: str, system_ctx: str = "", provider: str = None) -> dict:
    is_code = any(w in instruction.lower() for w in ["library", "framework", "github", "npm", "pypi", "package"])
    is_news = any(w in instruction.lower() for w in ["latest", "recent", "news", "today", "2024", "2025"])

    research_data = await full_research(instruction, include_github=is_code, include_news=is_news)
    context = research_data.get("combined", "")

    messages = [
        {"role": "system", "content": RESEARCH_SYS + (f"\n\n{system_ctx}" if system_ctx else "")},
        {"role": "user", "content": f"Research Task: {instruction}\n\nResearch Data:\n{context}\n\nAnalyze and provide comprehensive insights."}
    ]
    result = await llm_call(messages, task_type="research", provider=provider)
    return {"agent": "research_agent", "result": result["text"], "model": result["model"], "sources": list(research_data.keys())}


# ═══════════════════════════════════════ CODING AGENT ═══════════════════════════════════════
CODING_SYS = """You are Rem's Coding Director — an expert full-stack developer.
You write clean, production-ready, well-commented code.
Always use proper code blocks with language tags.
After writing code, explain what it does and how to run it.
For Python code that can be safely demonstrated, include example usage."""


async def coding_agent(instruction: str, system_ctx: str = "", provider: str = None, execute: bool = False) -> dict:
    messages = [
        {"role": "system", "content": CODING_SYS + (f"\n\n{system_ctx}" if system_ctx else "")},
        {"role": "user", "content": instruction}
    ]
    result = await llm_call(messages, task_type="coding", provider=provider)
    response_text = result["text"]
    execution_result = None

    if execute:
        code_match = re.search(r"```python\n([\s\S]*?)```", response_text)
        if code_match:
            code = code_match.group(1)
            execution_result = execute_python(code)
            if execution_result["executed"]:
                if execution_result["success"]:
                    response_text += f"\n\n**▶ Execution Output:**\n```\n{execution_result['output']}\n```"
                else:
                    response_text += f"\n\n**⚠ Execution Error:**\n```\n{execution_result['error']}\n```"

    return {"agent": "coding_agent", "result": response_text, "model": result["model"], "execution": execution_result}


# ═══════════════════════════════════════ PLANNING AGENT ═══════════════════════════════════════
PLANNING_SYS = """You are Rem's Planning Director — a strategic project manager and systems architect.
You break complex goals into actionable plans with clear timelines, priorities, and milestones.
Format: Use numbered lists, phases, timelines, and success metrics.
Always think: What's the minimum viable path? What are the risks? What can be automated?"""


async def planning_agent(instruction: str, goals: list = None, system_ctx: str = "", provider: str = None) -> dict:
    goals_ctx = ""
    if goals:
        active = [g["text"] for g in goals if not g.get("done")]
        if active:
            goals_ctx = f"\n\nUser's Active Goals:\n" + "\n".join(f"• {g}" for g in active)

    messages = [
        {"role": "system", "content": PLANNING_SYS + goals_ctx + (f"\n\n{system_ctx}" if system_ctx else "")},
        {"role": "user", "content": instruction}
    ]
    result = await llm_call(messages, task_type="planning", provider=provider)
    return {"agent": "planning_agent", "result": result["text"], "model": result["model"]}


# ═══════════════════════════════════════ AUTOMATION AGENT ═══════════════════════════════════════
AUTOMATION_SYS = """You are Rem's Automation Director — an expert in workflow automation, APIs, and no-code/low-code tools.
You design practical automation workflows using n8n, Zapier, Make, and custom APIs.
For each automation:
1. Describe the workflow with clear trigger → action → output steps
2. Provide the n8n/Zapier node configuration
3. Include any code nodes needed
4. Estimate time saved per week
5. List required API keys or permissions"""


async def automation_agent(instruction: str, system_ctx: str = "", provider: str = None) -> dict:
    messages = [
        {"role": "system", "content": AUTOMATION_SYS + (f"\n\n{system_ctx}" if system_ctx else "")},
        {"role": "user", "content": instruction}
    ]
    result = await llm_call(messages, task_type="automation", provider=provider)
    return {"agent": "automation_agent", "result": result["text"], "model": result["model"]}


# ═══════════════════════════════════════ CREATIVE AGENT ═══════════════════════════════════════
CREATIVE_SYS = """You are Rem's Creative Director — an expert content creator, copywriter, and media strategist.
You produce high-quality, engaging content for YouTube, blogs, social media, and marketing.
Your content is: Hook-driven, well-structured, SEO-aware, and platform-optimized.
Always include: Title options, hooks, full structure, key talking points, and CTAs."""


async def creative_agent(instruction: str, system_ctx: str = "", provider: str = None) -> dict:
    messages = [
        {"role": "system", "content": CREATIVE_SYS + (f"\n\n{system_ctx}" if system_ctx else "")},
        {"role": "user", "content": instruction}
    ]
    result = await llm_call(messages, task_type="creative", provider=provider)
    return {"agent": "creative_agent", "result": result["text"], "model": result["model"]}


# ═══════════════════════════════════════ AGENT REGISTRY ═══════════════════════════════════════
AGENT_REGISTRY = {
    "research_agent": research_agent,
    "coding_agent": coding_agent,
    "planning_agent": planning_agent,
    "automation_agent": automation_agent,
    "creative_agent": creative_agent,
}


async def run_agent(agent_name: str, instruction: str, system_ctx: str = "", provider: str = None, **kwargs) -> dict:
    """Run a specific agent by name."""
    agent_fn = AGENT_REGISTRY.get(agent_name)
    if not agent_fn:
        # Default to general LLM call
        result = await llm_call(
            [{"role": "system", "content": f"You are Rem, an advanced AI OS.\n{system_ctx}"}, {"role": "user", "content": instruction}],
            task_type="general", provider=provider
        )
        return {"agent": "general", "result": result["text"], "model": result["model"]}
    return await agent_fn(instruction, system_ctx=system_ctx, provider=provider, **kwargs)


async def run_agents_parallel(agent_tasks: list, system_ctx: str = "", provider: str = None) -> list[dict]:
    """Run multiple agents in parallel."""
    coroutines = [
        run_agent(task["agent"], task["instruction"], system_ctx=system_ctx, provider=provider)
        for task in agent_tasks
    ]
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    output = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            output.append({"agent": agent_tasks[i]["agent"], "result": f"Agent error: {str(result)}", "model": "error"})
        else:
            output.append(result)
    return output
