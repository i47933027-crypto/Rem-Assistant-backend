import asyncio
import json
from llm_router import llm_call


ORCHESTRATOR_SYSTEM = """You are Rem's Supreme Commander — the central orchestrator of a multi-agent AI OS.

Your job is to analyze tasks and produce a JSON execution plan.

AGENTS AVAILABLE:
- research_agent: web search, Wikipedia, GitHub, news analysis
- coding_agent: write code, debug, explain, optimize, execute Python
- planning_agent: break goals into tasks, create roadmaps, timelines
- automation_agent: design n8n/Zapier workflows, API integrations
- creative_agent: write scripts, blog posts, YouTube content, social media

RULES:
- Decompose complex tasks into subtasks assigned to specialist agents
- Simple tasks → single agent
- Complex tasks → multiple agents in parallel
- Always return valid JSON

OUTPUT FORMAT:
{
  "task_type": "coding|research|planning|automation|creative|general",
  "complexity": "simple|medium|complex",
  "agents": [
    {
      "agent": "agent_name",
      "role": "what this agent does in this task",
      "instruction": "specific instruction for this agent",
      "priority": 1
    }
  ],
  "synthesis_instruction": "how to combine agent results into final response"
}"""


async def orchestrate(task: str, context: str = "", provider: str = None) -> dict:
    """Analyze task and create execution plan."""
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {"role": "user", "content": f"Task: {task}\n\nContext: {context}\n\nCreate execution plan as JSON."}
    ]
    try:
        result = await llm_call(messages, task_type="fast", provider=provider)
        text = result["text"].strip()
        # Extract JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        plan = json.loads(text)
        return plan
    except Exception as e:
        # Fallback: simple single-agent plan
        task_lower = task.lower()
        if any(w in task_lower for w in ["code", "script", "program", "debug", "python", "javascript"]):
            agent = "coding_agent"
        elif any(w in task_lower for w in ["research", "find", "search", "analyze", "news"]):
            agent = "research_agent"
        elif any(w in task_lower for w in ["plan", "roadmap", "strategy", "goal"]):
            agent = "planning_agent"
        elif any(w in task_lower for w in ["automate", "workflow", "zapier", "n8n"]):
            agent = "automation_agent"
        elif any(w in task_lower for w in ["write", "blog", "youtube", "script", "content"]):
            agent = "creative_agent"
        else:
            agent = "coding_agent"  # default capable agent

        return {
            "task_type": "general",
            "complexity": "simple",
            "agents": [{"agent": agent, "role": "primary", "instruction": task, "priority": 1}],
            "synthesis_instruction": "Return the agent's response directly."
        }


async def synthesize(task: str, agent_results: list[dict], instruction: str, provider: str = None) -> str:
    """Combine multiple agent results into a coherent final response."""
    if len(agent_results) == 1:
        return agent_results[0].get("result", "")

    results_text = "\n\n".join(
        f"=== {r['agent'].upper()} RESULT ===\n{r['result']}"
        for r in agent_results
    )
    messages = [
        {"role": "system", "content": f"""You are Rem, an advanced AI OS. You have received results from multiple specialized agents.
Synthesize them into one coherent, comprehensive response.

Instruction: {instruction}

Be direct, professional, and structured. Use markdown formatting."""},
        {"role": "user", "content": f"Original Task: {task}\n\nAgent Results:\n{results_text}\n\nSynthesize into a unified response."}
    ]
    result = await llm_call(messages, task_type="synthesis", provider=provider)
    return result["text"]
