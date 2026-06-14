from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ai_tools import (
    image_stability, image_dalle, image_replicate, image_huggingface,
    video_kling, video_runway, video_pika,
    voice_elevenlabs, voice_list_elevenlabs,
    search_perplexity,
    call_together, call_mistral, call_cohere,
    get_available_tools, TOGETHER_MODELS,
)
from config import settings

router = APIRouter()


# ═══ STATUS ═══════════════════════════════════════════════════

@router.get("/status")
async def tools_status():
    """List all tools and their connection status."""
    tools = get_available_tools(settings)
    connected = [k for k, v in tools.items() if v["status"] == "connected"]
    return {
        "total": len(tools),
        "connected": len(connected),
        "connected_tools": connected,
        "tools": tools,
    }


# ═══ IMAGE GENERATION ════════════════════════════════════════

class ImageRequest(BaseModel):
    prompt: str
    provider: str = "huggingface"   # huggingface | stability | dalle | replicate
    negative: Optional[str] = ""
    width: int = 1024
    height: int = 1024
    model: Optional[str] = None

@router.post("/image")
async def generate_image(req: ImageRequest):
    try:
        if req.provider == "stability":
            return await image_stability(req.prompt, req.negative, req.width, req.height)
        elif req.provider == "dalle":
            return await image_dalle(req.prompt)
        elif req.provider == "replicate":
            return await image_replicate(req.prompt, req.model or "stability-ai/sdxl:39ed52f2")
        else:  # huggingface (free, no key needed)
            return await image_huggingface(req.prompt, req.model or "stabilityai/stable-diffusion-xl-base-1.0")
    except Exception as e:
        raise HTTPException(400, str(e))


# ═══ VIDEO GENERATION ════════════════════════════════════════

class VideoRequest(BaseModel):
    prompt: str
    provider: str = "kling"     # kling | runway | pika
    duration: int = 5
    aspect: str = "16:9"
    image_url: Optional[str] = None

@router.post("/video")
async def generate_video(req: VideoRequest):
    try:
        if req.provider == "kling":
            return await video_kling(req.prompt, req.duration, req.aspect)
        elif req.provider == "runway":
            return await video_runway(req.prompt, req.image_url, req.duration)
        elif req.provider == "pika":
            return await video_pika(req.prompt)
        else:
            raise HTTPException(400, f"Unknown video provider: {req.provider}")
    except Exception as e:
        raise HTTPException(400, str(e))


# ═══ VOICE ════════════════════════════════════════════════════

class VoiceRequest(BaseModel):
    text: str
    voice: str = "rachel"
    model: str = "eleven_monolingual_v1"

@router.post("/voice")
async def generate_voice(req: VoiceRequest):
    try:
        return await voice_elevenlabs(req.text, req.voice, req.model)
    except Exception as e:
        raise HTTPException(400, str(e))

@router.get("/voice/voices")
async def list_voices():
    try:
        return await voice_list_elevenlabs()
    except Exception as e:
        raise HTTPException(400, str(e))


# ═══ AI SEARCH ════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    model: str = "llama-3.1-sonar-large-128k-online"

@router.post("/search")
async def ai_search(req: SearchRequest):
    try:
        return await search_perplexity(req.query, req.model)
    except Exception as e:
        raise HTTPException(400, str(e))


# ═══ MULTI-MODEL LLM ══════════════════════════════════════════

class LLMRequest(BaseModel):
    messages: list
    provider: str = "together"     # together | mistral | cohere
    model: str = "llama-70b"
    max_tokens: int = 1500

@router.post("/llm")
async def multi_model_llm(req: LLMRequest):
    try:
        if req.provider == "together":
            return await call_together(req.messages, req.model, req.max_tokens)
        elif req.provider == "mistral":
            return await call_mistral(req.messages, req.model)
        elif req.provider == "cohere":
            prompt = req.messages[-1]["content"] if req.messages else ""
            return await call_cohere(prompt, req.model)
        raise HTTPException(400, f"Unknown LLM provider: {req.provider}")
    except Exception as e:
        raise HTTPException(400, str(e))

@router.get("/llm/models")
async def list_models():
    return {"together": TOGETHER_MODELS, "mistral": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "codestral-latest"], "cohere": ["command-r-plus", "command-r", "command"]}
