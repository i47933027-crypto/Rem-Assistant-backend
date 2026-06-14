"""
REM OS — AI TOOLS ENGINE
Integrates: Kling · Runway · Pika · Stability AI · DALL-E · Replicate
            ElevenLabs · HuggingFace · Perplexity · Together AI · Mistral · Cohere
"""

import httpx
import asyncio
import base64
from typing import Optional
from config import settings


# ═══════════════════════════════════════════════════════════════
# IMAGE GENERATION
# ═══════════════════════════════════════════════════════════════

async def image_stability(prompt: str, negative: str = "", width: int = 1024, height: int = 1024) -> dict:
    """Stability AI SDXL — free tier available."""
    if not settings.STABILITY_API_KEY:
        raise Exception("STABILITY_API_KEY not set")
    async with httpx.AsyncClient(timeout=60.0) as c:
        prompts = [{"text": prompt, "weight": 1.0}]
        if negative:
            prompts.append({"text": negative, "weight": -1.0})
        r = await c.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={"Authorization": f"Bearer {settings.STABILITY_API_KEY}", "Accept": "application/json"},
            json={"text_prompts": prompts, "cfg_scale": 7, "width": width, "height": height, "samples": 1, "steps": 30},
        )
        d = r.json()
        if "artifacts" in d:
            return {"success": True, "image_b64": d["artifacts"][0]["base64"], "seed": d["artifacts"][0].get("seed"), "provider": "stability_ai"}
        raise Exception(d.get("message", "Stability AI error"))


async def image_dalle(prompt: str, size: str = "1024x1024", quality: str = "standard") -> dict:
    """DALL-E 3 via OpenAI API."""
    if not settings.OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set")
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={"model": "dall-e-3", "prompt": prompt, "n": 1, "size": size, "quality": quality},
        )
        d = r.json()
        if "data" in d:
            return {"success": True, "url": d["data"][0]["url"], "revised_prompt": d["data"][0].get("revised_prompt"), "provider": "dalle3"}
        raise Exception(d.get("error", {}).get("message", "DALL-E error"))


async def image_replicate(prompt: str, model: str = "stability-ai/sdxl:39ed52f2319f9daed9d6d5a2f4b4b8d9bb04f6b9ab7b3c0e3f8b2c9e5d8a4f7a") -> dict:
    """Replicate — run any image model."""
    if not settings.REPLICATE_API_KEY:
        raise Exception("REPLICATE_API_KEY not set")
    async with httpx.AsyncClient(timeout=180.0) as c:
        r = await c.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {settings.REPLICATE_API_KEY}"},
            json={"version": model, "input": {"prompt": prompt, "num_outputs": 1}},
        )
        pred = r.json()
        pid = pred["id"]
        for _ in range(40):
            await asyncio.sleep(3)
            pr = await c.get(f"https://api.replicate.com/v1/predictions/{pid}", headers={"Authorization": f"Token {settings.REPLICATE_API_KEY}"})
            pd_ = pr.json()
            if pd_["status"] == "succeeded":
                out = pd_["output"]
                url = out[0] if isinstance(out, list) else out
                return {"success": True, "url": url, "provider": "replicate", "model": model}
            elif pd_["status"] == "failed":
                raise Exception(pd_.get("error", "Replicate failed"))
    raise Exception("Replicate timed out")


async def image_huggingface(prompt: str, model: str = "stabilityai/stable-diffusion-xl-base-1.0") -> dict:
    """HuggingFace Inference API — free models."""
    headers = {"Content-Type": "application/json"}
    if settings.HUGGINGFACE_API_KEY:
        headers["Authorization"] = f"Bearer {settings.HUGGINGFACE_API_KEY}"
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json={"inputs": prompt})
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            return {"success": True, "image_b64": base64.b64encode(r.content).decode(), "provider": "huggingface", "model": model}
        raise Exception(f"HuggingFace error: {r.text[:200]}")


# ═══════════════════════════════════════════════════════════════
# VIDEO GENERATION
# ═══════════════════════════════════════════════════════════════

async def video_kling(prompt: str, duration: int = 5, aspect: str = "16:9", mode: str = "std") -> dict:
    """Kling AI — text to video (Kuaishou)."""
    if not settings.KLING_API_KEY:
        raise Exception("KLING_API_KEY not set")
    async with httpx.AsyncClient(timeout=400.0) as c:
        r = await c.post(
            "https://api.klingai.com/v1/videos/text2video",
            headers={"Authorization": f"Bearer {settings.KLING_API_KEY}", "Content-Type": "application/json"},
            json={"model_name": "kling-v1", "prompt": prompt, "duration": str(duration), "aspect_ratio": aspect, "mode": mode},
        )
        d = r.json()
        if d.get("code") != 0:
            raise Exception(d.get("message", "Kling error"))
        task_id = d["data"]["task_id"]
        for _ in range(80):
            await asyncio.sleep(5)
            pr = await c.get(f"https://api.klingai.com/v1/videos/text2video/{task_id}", headers={"Authorization": f"Bearer {settings.KLING_API_KEY}"})
            pd_ = pr.json()
            status = pd_.get("data", {}).get("task_status", "")
            if status == "succeed":
                videos = pd_["data"].get("task_result", {}).get("videos", [])
                if videos:
                    return {"success": True, "url": videos[0]["url"], "duration": videos[0].get("duration"), "provider": "kling"}
            elif status == "failed":
                raise Exception(pd_.get("data", {}).get("task_status_msg", "Kling failed"))
    raise Exception("Kling AI timed out")


async def video_runway(prompt: str, image_url: Optional[str] = None, duration: int = 5) -> dict:
    """Runway ML Gen-3 — text/image to video."""
    if not settings.RUNWAY_API_KEY:
        raise Exception("RUNWAY_API_KEY not set")
    headers = {"Authorization": f"Bearer {settings.RUNWAY_API_KEY}", "X-Runway-Version": "2024-11-06"}
    payload = {"model": "gen3a_turbo", "promptText": prompt, "duration": duration, "ratio": "1280:768"}
    if image_url:
        payload["promptImage"] = image_url
    async with httpx.AsyncClient(timeout=400.0) as c:
        r = await c.post("https://api.runwayml.com/v1/image_to_video", headers=headers, json=payload)
        d = r.json()
        task_id = d.get("id")
        if not task_id:
            raise Exception(d.get("error", "Runway error"))
        for _ in range(80):
            await asyncio.sleep(5)
            pr = await c.get(f"https://api.runwayml.com/v1/tasks/{task_id}", headers=headers)
            pd_ = pr.json()
            if pd_.get("status") == "SUCCEEDED":
                return {"success": True, "url": pd_["output"][0], "provider": "runway"}
            elif pd_.get("status") in ("FAILED", "CANCELLED"):
                raise Exception("Runway ML generation failed")
    raise Exception("Runway ML timed out")


async def video_pika(prompt: str, style: str = "default") -> dict:
    """Pika Labs — text to video."""
    if not settings.PIKA_API_KEY:
        raise Exception("PIKA_API_KEY not set")
    async with httpx.AsyncClient(timeout=300.0) as c:
        r = await c.post(
            "https://api.pika.art/generate",
            headers={"Authorization": f"Bearer {settings.PIKA_API_KEY}", "Content-Type": "application/json"},
            json={"prompt": prompt, "style": style, "aspectRatio": "16:9"},
        )
        d = r.json()
        job_id = d.get("jobId") or d.get("id")
        if not job_id:
            raise Exception(d.get("message", "Pika error"))
        for _ in range(60):
            await asyncio.sleep(5)
            pr = await c.get(f"https://api.pika.art/jobs/{job_id}", headers={"Authorization": f"Bearer {settings.PIKA_API_KEY}"})
            pd_ = pr.json()
            if pd_.get("status") == "finished":
                return {"success": True, "url": pd_.get("resultUrl") or pd_.get("video", {}).get("url"), "provider": "pika"}
            elif pd_.get("status") == "failed":
                raise Exception("Pika failed")
    raise Exception("Pika timed out")


# ═══════════════════════════════════════════════════════════════
# VOICE / AUDIO
# ═══════════════════════════════════════════════════════════════

ELEVENLABS_VOICES = {
    "rachel":  "21m00Tcm4TlvDq8ikWAM",
    "domi":    "AZnzlk1XvdvUeBnXmlld",
    "bella":   "EXAVITQu4vr4xnSDxMaL",
    "josh":    "TxGEqnHWrfWFTfGW9XjX",
    "arnold":  "VR6AewLTigWG4xSOukaG",
    "adam":    "pNInz6obpgDQGcFmaJgB",
    "sam":     "yoZ06aMxZJJ28mfd3POQ",
}

async def voice_elevenlabs(text: str, voice: str = "rachel", model: str = "eleven_monolingual_v1") -> dict:
    """ElevenLabs — realistic text to speech. Free tier: 10,000 chars/month."""
    if not settings.ELEVENLABS_API_KEY:
        raise Exception("ELEVENLABS_API_KEY not set")
    voice_id = ELEVENLABS_VOICES.get(voice.lower(), voice)
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": settings.ELEVENLABS_API_KEY, "Accept": "audio/mpeg", "Content-Type": "application/json"},
            json={"text": text, "model_id": model, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        )
        if r.status_code == 200:
            return {"success": True, "audio_b64": base64.b64encode(r.content).decode(), "format": "mp3", "provider": "elevenlabs", "voice": voice}
        raise Exception(f"ElevenLabs: {r.text[:200]}")


async def voice_list_elevenlabs() -> list:
    """List available ElevenLabs voices."""
    if not settings.ELEVENLABS_API_KEY:
        return list(ELEVENLABS_VOICES.keys())
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": settings.ELEVENLABS_API_KEY})
        d = r.json()
        return [{"id": v["voice_id"], "name": v["name"], "category": v.get("category", "")} for v in d.get("voices", [])]


# ═══════════════════════════════════════════════════════════════
# AI SEARCH
# ═══════════════════════════════════════════════════════════════

async def search_perplexity(query: str, model: str = "llama-3.1-sonar-large-128k-online") -> dict:
    """Perplexity AI — real-time web search with AI answer + citations."""
    if not settings.PERPLEXITY_API_KEY:
        raise Exception("PERPLEXITY_API_KEY not set")
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": query}], "max_tokens": 1500, "return_citations": True},
        )
        d = r.json()
        if "choices" in d:
            return {"success": True, "answer": d["choices"][0]["message"]["content"], "citations": d.get("citations", []), "provider": "perplexity"}
        raise Exception(d.get("error", {}).get("message", "Perplexity error"))


# ═══════════════════════════════════════════════════════════════
# MULTI-MODEL AI
# ═══════════════════════════════════════════════════════════════

TOGETHER_MODELS = {
    "llama-405b":   "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "llama-70b":    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "mistral-7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral":      "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "qwen-72b":     "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "deepseek-r1":  "deepseek-ai/DeepSeek-R1",
    "gemma-27b":    "google/gemma-2-27b-it",
    "dbrx":         "databricks/dbrx-instruct",
}

async def call_together(messages: list, model_key: str = "llama-70b", max_tokens: int = 1500) -> dict:
    """Together AI — access 100+ open source models with free $1 credit."""
    if not settings.TOGETHER_API_KEY:
        raise Exception("TOGETHER_API_KEY not set")
    model = TOGETHER_MODELS.get(model_key, model_key)
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.TOGETHER_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.72},
        )
        d = r.json()
        if "choices" in d:
            return {"success": True, "text": d["choices"][0]["message"]["content"], "model": model, "provider": "together"}
        raise Exception(str(d.get("error", d))[:200])


async def call_mistral(messages: list, model: str = "mistral-large-latest") -> dict:
    """Mistral AI — fast European AI, free tier available."""
    if not settings.MISTRAL_API_KEY:
        raise Exception("MISTRAL_API_KEY not set")
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.MISTRAL_API_KEY}"},
            json={"model": model, "messages": messages, "max_tokens": 1500},
        )
        d = r.json()
        if "choices" in d:
            return {"success": True, "text": d["choices"][0]["message"]["content"], "model": model, "provider": "mistral"}
        raise Exception(str(d.get("error", d))[:200])


async def call_cohere(prompt: str, model: str = "command-r-plus") -> dict:
    """Cohere — RAG and enterprise AI, free trial available."""
    if not settings.COHERE_API_KEY:
        raise Exception("COHERE_API_KEY not set")
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(
            "https://api.cohere.com/v1/chat",
            headers={"Authorization": f"Bearer {settings.COHERE_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "message": prompt, "max_tokens": 1500},
        )
        d = r.json()
        if "text" in d:
            return {"success": True, "text": d["text"], "model": model, "provider": "cohere"}
        raise Exception(str(d.get("message", d))[:200])


# ═══════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ═══════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    # Image
    "stability":   {"name": "Stability AI",  "type": "image",  "icon": "🎨", "free": True,  "fn": image_stability,   "key": "STABILITY_API_KEY",   "url": "https://stability.ai",           "desc": "SDXL image generation"},
    "dalle":       {"name": "DALL-E 3",       "type": "image",  "icon": "🖼",  "free": False, "fn": image_dalle,       "key": "OPENAI_API_KEY",       "url": "https://openai.com",             "desc": "OpenAI image generation"},
    "replicate":   {"name": "Replicate",      "type": "image",  "icon": "⚡",  "free": False, "fn": image_replicate,   "key": "REPLICATE_API_KEY",    "url": "https://replicate.com",          "desc": "Run any AI model"},
    "huggingface": {"name": "HuggingFace",    "type": "image",  "icon": "🤗",  "free": True,  "fn": image_huggingface, "key": "HUGGINGFACE_API_KEY",  "url": "https://huggingface.co",         "desc": "Free open models"},
    # Video
    "kling":       {"name": "Kling AI",       "type": "video",  "icon": "🎬",  "free": False, "fn": video_kling,       "key": "KLING_API_KEY",        "url": "https://klingai.com",            "desc": "Kling text-to-video"},
    "runway":      {"name": "Runway ML",      "type": "video",  "icon": "🎥",  "free": False, "fn": video_runway,      "key": "RUNWAY_API_KEY",       "url": "https://runwayml.com",           "desc": "Gen-3 video generation"},
    "pika":        {"name": "Pika Labs",      "type": "video",  "icon": "✨",  "free": False, "fn": video_pika,        "key": "PIKA_API_KEY",         "url": "https://pika.art",               "desc": "Pika video generation"},
    # Voice
    "elevenlabs":  {"name": "ElevenLabs",     "type": "voice",  "icon": "🎙",  "free": True,  "fn": voice_elevenlabs,  "key": "ELEVENLABS_API_KEY",   "url": "https://elevenlabs.io",          "desc": "10k chars/month free"},
    # Search
    "perplexity":  {"name": "Perplexity",     "type": "search", "icon": "🔍",  "free": False, "fn": search_perplexity, "key": "PERPLEXITY_API_KEY",   "url": "https://perplexity.ai",          "desc": "AI search + citations"},
    # LLM
    "together":    {"name": "Together AI",    "type": "llm",    "icon": "🧠",  "free": True,  "fn": call_together,     "key": "TOGETHER_API_KEY",     "url": "https://api.together.xyz",       "desc": "$1 free credit, 100+ models"},
    "mistral":     {"name": "Mistral AI",     "type": "llm",    "icon": "🌊",  "free": True,  "fn": call_mistral,      "key": "MISTRAL_API_KEY",      "url": "https://mistral.ai",             "desc": "Fast EU AI, free tier"},
    "cohere":      {"name": "Cohere",         "type": "llm",    "icon": "🌐",  "free": True,  "fn": call_cohere,       "key": "COHERE_API_KEY",       "url": "https://cohere.com",             "desc": "Command R+ free trial"},
}


def get_available_tools(settings_obj) -> dict:
    """Return only tools with API keys configured."""
    available = {}
    for key, tool in TOOL_REGISTRY.items():
        env_key = tool["key"]
        if getattr(settings_obj, env_key, None):
            available[key] = {**tool, "status": "connected"}
        else:
            available[key] = {**tool, "status": "disconnected"}
    return available
