from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from database import init_db, IS_POSTGRES
from memory_core import reload_chromadb_from_db
from routers.chat import router as chat_router
from routers.all_routers import memory_router, goals_router, system_router
from routers.connectors import router as connectors_router
from routers.n8n import router as n8n_router
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Rem OS Backend — Starting...")
    await init_db()
    db_type = "Supabase PostgreSQL ☁" if IS_POSTGRES else "SQLite (local)"
    print(f"✅ Database: {db_type}")
    # Reload memories from DB into ChromaDB for semantic search
    await reload_chromadb_from_db()
    print(f"✅ AI Brain: {settings.DEFAULT_PROVIDER}")
    print(f"✅ Web Search: DuckDuckGo online")
    print(f"✅ GitHub Search: online")
    n8n = "✅ Connected" if settings.N8N_URL else "⚠️  Not configured"
    print(f"   N8N: {n8n}")
    print("✅ Rem OS v2.0 ready — all 11 modules active\n")
    yield
    print("⏹  Rem OS shutting down")


app = FastAPI(title="Rem OS Backend", version="2.0.0", lifespan=lifespan)

origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router,       prefix="/api/chat",       tags=["💬 Chat"])
app.include_router(memory_router,     prefix="/api/memory",     tags=["🧠 Memory"])
app.include_router(goals_router,      prefix="/api/goals",      tags=["🎯 Goals"])
app.include_router(system_router,     prefix="/api/system",     tags=["⚙ System"])
app.include_router(connectors_router, prefix="/api/connectors", tags=["🔌 Connectors"])
app.include_router(n8n_router,        prefix="/api/n8n",        tags=["⚡ N8N"])


@app.get("/")
async def root():
    return {
        "name":           "Rem OS Backend",
        "version":        "2.0.0",
        "status":         "online",
        "database":       "supabase" if IS_POSTGRES else "sqlite",
        "n8n_configured": bool(settings.N8N_URL),
        "docs":           "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
