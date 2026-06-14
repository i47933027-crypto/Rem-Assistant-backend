from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── AI Brain ─────────────────────────────────────────────
    GROQ_API_KEY:        Optional[str] = None
    GEMINI_API_KEY:      Optional[str] = None
    OLLAMA_HOST:         str           = "http://localhost:11434"
    DEFAULT_PROVIDER:    str           = "groq"

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL:        str           = "sqlite+aiosqlite:///./rem_os.db"
    SUPABASE_DB_URL:     Optional[str] = None   # overrides DATABASE_URL
    CHROMA_PATH:         str           = "./chroma_db"

    # ── Server ───────────────────────────────────────────────
    HOST:                str           = "0.0.0.0"
    PORT:                int           = 8000
    CORS_ORIGINS:        str           = "*"

    # ── N8N ──────────────────────────────────────────────────
    N8N_URL:             Optional[str] = None
    N8N_API_KEY:         Optional[str] = None

    # ── Connectors ───────────────────────────────────────────
    GITHUB_TOKEN:        Optional[str] = None
    NOTION_API_KEY:      Optional[str] = None
    TELEGRAM_BOT_TOKEN:  Optional[str] = None

    # ── AI Tools (optional) ───────────────────────────────────
    STABILITY_API_KEY:   Optional[str] = None
    OPENAI_API_KEY:      Optional[str] = None
    REPLICATE_API_KEY:   Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    KLING_API_KEY:       Optional[str] = None
    RUNWAY_API_KEY:      Optional[str] = None
    PIKA_API_KEY:        Optional[str] = None
    ELEVENLABS_API_KEY:  Optional[str] = None
    PERPLEXITY_API_KEY:  Optional[str] = None
    TOGETHER_API_KEY:    Optional[str] = None
    MISTRAL_API_KEY:     Optional[str] = None
    COHERE_API_KEY:      Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Model routing
GROQ_MODEL_ROUTES = {
    "coding":     "deepseek-r1-distill-llama-70b",
    "research":   "llama-3.3-70b-versatile",
    "planning":   "llama-3.3-70b-versatile",
    "automation": "llama-3.3-70b-versatile",
    "creative":   "llama-3.3-70b-versatile",
    "fast":       "llama-3.1-8b-instant",
    "general":    "llama-3.3-70b-versatile",
    "synthesis":  "llama-3.3-70b-versatile",
}
GEMINI_MODEL_ROUTES = {k: "gemini-1.5-flash" for k in GROQ_MODEL_ROUTES}
GEMINI_MODEL_ROUTES.update({"research": "gemini-1.5-pro", "planning": "gemini-1.5-pro"})
OLLAMA_MODEL_ROUTES = {k: "llama3.2" for k in GROQ_MODEL_ROUTES}
OLLAMA_MODEL_ROUTES["coding"] = "deepseek-r1"
