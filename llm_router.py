import httpx
from typing import Optional
from config import settings, GROQ_MODEL_ROUTES, GEMINI_MODEL_ROUTES, OLLAMA_MODEL_ROUTES


async def call_groq(messages: list, task_type: str = "general", override_model: Optional[str] = None) -> dict:
    model = override_model or GROQ_MODEL_ROUTES.get(task_type, "llama-3.3-70b-versatile")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            json={"model": model, "messages": messages, "max_tokens": 2000, "temperature": 0.72},
        )
        data = resp.json()
        if "error" in data:
            raise Exception(data["error"]["message"])
        return {"text": data["choices"][0]["message"]["content"], "model": model, "provider": "groq"}


async def call_gemini(messages: list, task_type: str = "general", override_model: Optional[str] = None) -> dict:
    model = override_model or GEMINI_MODEL_ROUTES.get(task_type, "gemini-1.5-flash")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    body = {"contents": contents, "generationConfig": {"maxOutputTokens": 2000, "temperature": 0.72}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=body)
        data = resp.json()
        if "error" in data:
            raise Exception(data["error"]["message"])
        return {"text": data["candidates"][0]["content"]["parts"][0]["text"], "model": model, "provider": "gemini"}


async def call_ollama(messages: list, task_type: str = "general", override_model: Optional[str] = None) -> dict:
    model = override_model or OLLAMA_MODEL_ROUTES.get(task_type, "llama3.2")
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.OLLAMA_HOST}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        data = resp.json()
        if "error" in data:
            raise Exception(data["error"])
        return {"text": data["message"]["content"], "model": model, "provider": "ollama"}


async def llm_call(
    messages: list,
    task_type: str = "general",
    provider: Optional[str] = None,
    override_model: Optional[str] = None,
) -> dict:
    """Route LLM call to best available provider."""
    p = provider or settings.DEFAULT_PROVIDER

    # Primary call
    try:
        if p == "groq" and settings.GROQ_API_KEY:
            return await call_groq(messages, task_type, override_model)
        elif p == "gemini" and settings.GEMINI_API_KEY:
            return await call_gemini(messages, task_type, override_model)
        elif p == "ollama":
            return await call_ollama(messages, task_type, override_model)
    except Exception as e:
        print(f"Primary provider {p} failed: {e}")

    # Fallback chain
    for fallback in ["groq", "gemini", "ollama"]:
        if fallback == p:
            continue
        try:
            if fallback == "groq" and settings.GROQ_API_KEY:
                return await call_groq(messages, task_type)
            elif fallback == "gemini" and settings.GEMINI_API_KEY:
                return await call_gemini(messages, task_type)
            elif fallback == "ollama":
                return await call_ollama(messages, task_type)
        except Exception as e:
            print(f"Fallback {fallback} also failed: {e}")

    raise Exception("All LLM providers failed. Check your API keys.")
