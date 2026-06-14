from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pydantic import BaseModel
from typing import Optional
from database import get_db, Memory, Goal, OptimizationLog, DigitalTwin
from memory_core import save_memory, delete_memory, search_memories

# ═══════════════════ MEMORY ROUTER ═══════════════════
memory_router = APIRouter()

class MemoryCreate(BaseModel):
    fact: str
    category: str = "general"
    importance: int = 5

@memory_router.get("")
async def list_memories(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Memory).order_by(Memory.importance.desc(), Memory.created_at.desc()))
    mems = result.scalars().all()
    return [{"id": m.id, "fact": m.fact, "category": m.category, "importance": m.importance, "created_at": m.created_at} for m in mems]

@memory_router.post("")
async def add_memory(body: MemoryCreate, db: AsyncSession = Depends(get_db)):
    await save_memory(body.fact, body.category, body.importance, db)
    return {"status": "saved", "fact": body.fact}

@memory_router.delete("/{memory_id}")
async def remove_memory(memory_id: int, db: AsyncSession = Depends(get_db)):
    ok = await delete_memory(memory_id, db)
    if not ok:
        raise HTTPException(404, "Memory not found")
    return {"status": "deleted"}

@memory_router.delete("")
async def clear_memories(db: AsyncSession = Depends(get_db)):
    await db.execute(delete(Memory))
    await db.commit()
    return {"status": "cleared"}

@memory_router.get("/search/{query}")
async def semantic_search(query: str):
    results = await search_memories(query)
    return {"query": query, "results": results}


# ═══════════════════ GOALS ROUTER ═══════════════════
goals_router = APIRouter()

class GoalCreate(BaseModel):
    text: str
    priority: int = 5

class GoalUpdate(BaseModel):
    text: Optional[str] = None
    done: Optional[bool] = None
    priority: Optional[int] = None

@goals_router.get("")
async def list_goals(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Goal).order_by(Goal.priority.desc(), Goal.created_at.desc()))
    return [{"id": g.id, "text": g.text, "done": g.done, "priority": g.priority, "created_at": g.created_at} for g in result.scalars().all()]

@goals_router.post("")
async def create_goal(body: GoalCreate, db: AsyncSession = Depends(get_db)):
    goal = Goal(text=body.text, priority=body.priority)
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    return {"id": goal.id, "text": goal.text, "done": goal.done}

@goals_router.put("/{goal_id}")
async def update_goal(goal_id: int, body: GoalUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Goal).where(Goal.id == goal_id))
    goal = result.scalar_one_or_none()
    if not goal:
        raise HTTPException(404, "Goal not found")
    if body.text is not None:
        goal.text = body.text
    if body.done is not None:
        goal.done = body.done
    if body.priority is not None:
        goal.priority = body.priority
    await db.commit()
    return {"id": goal.id, "text": goal.text, "done": goal.done}

@goals_router.delete("/{goal_id}")
async def delete_goal(goal_id: int, db: AsyncSession = Depends(get_db)):
    await db.execute(delete(Goal).where(Goal.id == goal_id))
    await db.commit()
    return {"status": "deleted"}


# ═══════════════════ SYSTEM ROUTER ═══════════════════
system_router = APIRouter()

@system_router.get("/status")
async def system_status(db: AsyncSession = Depends(get_db)):
    mem_count = (await db.execute(select(Memory))).scalars().all()
    goal_count = (await db.execute(select(Goal).where(Goal.done == False))).scalars().all()
    logs = (await db.execute(select(OptimizationLog).order_by(OptimizationLog.created_at.desc()).limit(10))).scalars().all()
    return {
        "status": "online",
        "version": "2.0.0",
        "modules": 11,
        "memories": len(mem_count),
        "active_goals": len(goal_count),
        "recent_logs": [{"event": l.event, "module": l.module, "time": l.created_at} for l in logs],
    }

@system_router.get("/logs")
async def get_logs(limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(OptimizationLog).order_by(OptimizationLog.created_at.desc()).limit(limit))
    logs = result.scalars().all()
    return [{"event": l.event, "module": l.module, "time": l.created_at} for l in logs]
