"""
REM OS — MEMORY CORE
Supabase PostgreSQL = persistent storage (survives restarts)
ChromaDB = fast in-memory semantic search (rebuilt from DB on startup)
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from database import Memory, OptimizationLog, AsyncSessionLocal
from config import settings
import re


# ── ChromaDB (in-memory for speed, rebuilt from DB on startup) ──
chroma_client = chromadb.Client()  # purely in-memory
memory_collection = chroma_client.get_or_create_collection(
    name="rem_memories",
    metadata={"hnsw:space": "cosine"}
)


async def reload_chromadb_from_db():
    """On startup: load all memories from PostgreSQL into ChromaDB."""
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Memory).order_by(Memory.created_at))
            memories = result.scalars().all()
            if not memories:
                print("◈ Memory Core: No stored memories found")
                return
            # Clear and reload
            try:
                memory_collection.delete(where={"db_id": {"$gte": 0}})
            except Exception:
                pass
            for m in memories:
                try:
                    memory_collection.upsert(
                        documents=[m.fact],
                        ids=[f"mem_{m.id}"],
                        metadatas=[{"category": m.category, "importance": m.importance, "db_id": m.id}]
                    )
                except Exception:
                    pass
            print(f"◈ Memory Core: Loaded {len(memories)} memories from database into ChromaDB")
    except Exception as e:
        print(f"◈ Memory Core: Could not reload from DB ({e}) — starting fresh")


def extract_facts(text: str) -> list[dict]:
    """Extract structured facts from user message."""
    facts = []
    patterns = [
        (r"(?:my name is|i'm called|call me)\s+([A-Z][a-z]+)", "identity", 10),
        (r"i (?:prefer|like|love|use|always use)\s+([^.!?,]{4,50})", "preference", 8),
        (r"(?:working on|building|developing|creating)\s+([^.!?,]{4,60})", "project", 9),
        (r"(?:using|with|in)\s+(python|javascript|typescript|react|node|rust|go|java|swift|php|flutter)", "tech_stack", 7),
        (r"(?:my goal is|i want to|i'm trying to|i plan to)\s+([^.!?]{6,80})", "goal", 9),
        (r"i (?:work at|work for|am at)\s+([^.!?,]{4,50})", "workplace", 8),
        (r"i am a\s+([^.!?,]{4,40})", "role", 7),
        (r"i live in\s+([^.!?,]{4,40})", "location", 6),
    ]
    for pattern, category, importance in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            facts.append({"fact": match.group(0).strip(), "category": category, "importance": importance})
    return facts


async def save_memory(fact: str, category: str, importance: int, db: AsyncSession) -> None:
    """Save memory to PostgreSQL (persistent) and ChromaDB (fast search)."""
    mem = Memory(fact=fact, category=category, importance=importance)
    db.add(mem)
    await db.commit()
    await db.refresh(mem)
    # Also index in ChromaDB
    try:
        memory_collection.upsert(
            documents=[fact],
            ids=[f"mem_{mem.id}"],
            metadatas=[{"category": category, "importance": importance, "db_id": mem.id}]
        )
    except Exception:
        pass


async def search_memories(query: str, n_results: int = 8) -> list[str]:
    """Fast semantic search using ChromaDB."""
    try:
        count = memory_collection.count()
        if count == 0:
            return []
        result = memory_collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        return result["documents"][0] if result["documents"] else []
    except Exception:
        return []


async def get_all_memories(db: AsyncSession) -> list[Memory]:
    result = await db.execute(select(Memory).order_by(Memory.importance.desc(), Memory.created_at.desc()))
    return result.scalars().all()


async def delete_memory(memory_id: int, db: AsyncSession) -> bool:
    result = await db.execute(select(Memory).where(Memory.id == memory_id))
    mem = result.scalar_one_or_none()
    if not mem:
        return False
    await db.execute(delete(Memory).where(Memory.id == memory_id))
    await db.commit()
    try:
        memory_collection.delete(ids=[f"mem_{memory_id}"])
    except Exception:
        pass
    return True


async def log_optimization(event: str, module: str, db: AsyncSession) -> None:
    log = OptimizationLog(event=event, module=module)
    db.add(log)
    await db.commit()


async def build_memory_context(query: str, db: AsyncSession) -> str:
    """Build memory context string for LLM system prompt."""
    relevant = await search_memories(query, n_results=6)
    all_mems = await get_all_memories(db)
    important = [m.fact for m in all_mems if m.importance >= 8][:4]
    combined = list(dict.fromkeys(important + relevant))[:10]
    if not combined:
        return ""
    return "[MEMORY CORE — Active Context]\n" + "\n".join(f"{i+1}. {f}" for i, f in enumerate(combined))
