import base64
import os
import time
import requests
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Depends
import uvicorn
from typing import List, Optional, Dict, Any
import random
import asyncio
from contextlib import asynccontextmanager
import uuid
from datetime import datetime, timedelta
import json

# Enhanced imports for new features
from prompts import PROMPTS, NanoBananaPrompts, PromptConfig, QualityLevel, StylePreset
import cachetools
from cachetools import TTLCache
from pydantic import BaseModel, Field
import aiofiles
from pathlib import Path

# === Configuration ===
class Config:
    """Configuration management"""
    def __init__(self):
        self.API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1.5"))
        self.MAX_AD_VARIATIONS = int(os.getenv("MAX_AD_VARIATIONS", "4"))
        self.MAX_SCENE_VARIATIONS = int(os.getenv("MAX_SCENE_VARIATIONS", "6"))
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10_485_760"))  # 10MB
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
        self.RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour

config = Config()

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nano_banana.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nano_banana")

# === Cache Setup ===
# Cache for API responses to avoid duplicate processing
response_cache = TTLCache(maxsize=1000, ttl=config.CACHE_TTL)

# Rate limiting storage
rate_limit_cache = TTLCache(maxsize=10000, ttl=config.RATE_LIMIT_WINDOW)

# === Enhanced Prompt Manager ===
prompt_manager = NanoBananaPrompts()

# === Request Models ===
class GenerationRequest(BaseModel):
    """Base model for generation requests"""
    api_key: str = Field(..., description="Gemini API Key")
    prompt: str = Field(..., description="User prompt for generation")
    quality: Optional[str] = Field("standard", description="Quality level")
    style: Optional[str] = Field(None, description="Style preset")
    enhance_creativity: Optional[bool] = Field(False, description="Enhance creativity")

class ImageResponse(BaseModel):
    """Standard image response model"""
    image: str = Field(..., description="Base64 encoded image")
    mime: str = Field(..., description="MIME type of image")
    request_id: str = Field(..., description="Unique request identifier")
    processing_time: float = Field(..., description="Processing time in seconds")

class BatchResponse(BaseModel):
    """Response model for batch operations"""
    results: List[ImageResponse]
    total_generated: int
    request_id: str

# === Utility Functions ===
def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{uuid.uuid4().hex[:12]}"

def check_rate_limit(api_key: str) -> bool:
    """Check if API key has exceeded rate limit"""
    key = f"rate_limit_{api_key}"
    if key in rate_limit_cache:
        rate_limit_cache[key] += 1
        if rate_limit_cache[key] > config.RATE_LIMIT_REQUESTS:
            return False
    else:
        rate_limit_cache[key] = 1
    return True

def get_cache_key(operation: str, prompt: str, images_data: List[str] = None) -> str:
    """Generate cache key for request"""
    key_parts = [operation, prompt]
    if images_data:
        for img in images_data[:3]:  # Use first few images for key
            key_parts.append(img[:100])  # Use partial image data
    return hash(''.join(key_parts))

async def validate_file_size(file: UploadFile) -> bool:
    """Validate uploaded file size"""
    try:
        # Get file size by seeking to end
        await file.seek(0, 2)  # Seek to end
        size = await file.tell()
        await file.seek(0)  # Reset to beginning
        return size <= config.MAX_FILE_SIZE
    except Exception:
        return False

def b64encode_file(file: UploadFile) -> tuple[str, str]:
    """Encode uploaded file to base64 with validation"""
    try:
        data = file.file.read()
        if len(data) > config.MAX_FILE_SIZE:
            raise HTTPException(413, f"File too large. Max size: {config.MAX_FILE_SIZE // 1024 // 1024}MB")
        
        mime = file.content_type or "image/png"
        if mime not in ["image/png", "image/jpeg", "image/jpg", "image/webp"]:
            raise HTTPException(415, f"Unsupported file type: {mime}")
        
        return base64.b64encode(data).decode('utf-8'), mime
    except Exception as e:
        logger.error(f"Error encoding file: {str(e)}")
        raise HTTPException(500, "Error processing uploaded file")

def call_nano_banana(
    api_key: str, 
    prompt: str, 
    images: List[dict] = None, 
    retries: int = None,
    backoff: float = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Enhanced API call with better error handling and retry logic
    """
    retries = retries or config.MAX_RETRIES
    backoff = backoff or config.BACKOFF_BASE
    
    parts = [{'text': prompt}]
    if images:
        for img in images:
            parts.append({'inlineData': {'data': img['data'], 'mimeType': img['mime']}})

    payload = {'contents': [{'parts': parts}]}
    attempt = 0
    
    while attempt <= retries:
        try:
            start_time = time.time()
            response = requests.post(
                f"{config.API_URL}?key={api_key}", 
                json=payload, 
                headers={"Content-Type": "application/json"},
                timeout=config.REQUEST_TIMEOUT
            )
            processing_time = time.time() - start_time
            logger.info(f"API call completed in {processing_time:.2f}s, status: {response.status_code}")

            if response.status_code == 429:
                # Rate limited - exponential backoff
                if attempt == retries:
                    return None, "Rate limit exceeded after retries"
                sleep_for = backoff ** attempt + random.uniform(0, 1)
                logger.warning(f"Rate limited, retrying in {sleep_for:.2f}s")
                time.sleep(sleep_for)
                attempt += 1
                continue
                
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return None, error_msg

            data = response.json()
            
            # Enhanced response parsing
            candidates = data.get('candidates', [])
            if not candidates:
                return None, "No candidates in response"
                
            content = candidates[0].get('content', {})
            parts_out = content.get('parts', [])
            
            for part in parts_out:
                if 'inlineData' in part:
                    inline_data = part['inlineData']
                    return inline_data['data'], inline_data.get('mimeType', 'image/png')
            
            # Check for safety filters
            safety_ratings = candidates[0].get('safetyRatings', [])
            if safety_ratings:
                blocked = any(rating.get('blocked', False) for rating in safety_ratings)
                if blocked:
                    return None, "Content blocked by safety filters"
            
            return None, "No image data in response"
            
        except requests.exceptions.Timeout:
            error_msg = "API request timeout"
            logger.error(error_msg)
            if attempt == retries:
                return None, error_msg
            attempt += 1
            time.sleep(backoff ** attempt)
            
        except requests.exceptions.ConnectionError:
            error_msg = "Connection error to API"
            logger.error(error_msg)
            if attempt == retries:
                return None, error_msg
            attempt += 1
            time.sleep(backoff ** attempt)
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            if attempt == retries:
                return None, error_msg
            attempt += 1
            time.sleep(backoff ** attempt)
    
    return None, "Exhausted all retry attempts"

# === Background Tasks ===
async def log_usage(api_key: str, endpoint: str, success: bool, processing_time: float):
    """Log usage statistics"""
    try:
        # In production, you might want to store this in a database
        logger.info(f"Usage - API Key: {api_key[:8]}..., Endpoint: {endpoint}, "
                   f"Success: {success}, Time: {processing_time:.2f}s")
    except Exception as e:
        logger.error(f"Error logging usage: {str(e)}")

# === FastAPI Application ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Nano Banana Studio API starting up...")
    yield
    # Shutdown
    logger.info("🛑 Nano Banana Studio API shutting down...")

app = FastAPI(
    title="Nano Banana Studio API",
    description="Enhanced AI Image Generation and Editing Platform",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Enhanced Endpoints ===
@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "cache_size": len(response_cache),
        "rate_limits_active": len(rate_limit_cache)
    }

@app.post("/generate", response_model=ImageResponse)
async def generate_image(
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    prompt: str = Form(...),
    quality: str = Form("standard"),
    style: Optional[str] = Form(None),
    enhance_creativity: bool = Form(False)
):
    """Enhanced image generation with quality and style controls"""
    start_time = time.time()
    request_id = generate_request_id()
    
    try:
        # Rate limiting
        if not check_rate_limit(api_key):
            raise HTTPException(429, "Rate limit exceeded")
        
        # Validate inputs
        if not prompt.strip():
            raise HTTPException(400, "Prompt cannot be empty")
        
        # Enhanced prompt generation
        prompt_config = PromptConfig(
            quality=QualityLevel(quality),
            style=StylePreset(style) if style else None,
            enhance_creativity=enhance_creativity
        )
        
        enhanced_prompts = prompt_manager.get_enhanced_prompt(
            "generate_image", 
            prompt, 
            prompt_config
        )
        system_prompt = random.choice(enhanced_prompts)
        full_prompt = f"{system_prompt} USER PROMPT: {prompt}"
        
        # Check cache
        cache_key = get_cache_key("generate", full_prompt)
        if cache_key in response_cache:
            logger.info(f"Cache hit for request {request_id}")
            img_b64, mime = response_cache[cache_key]
        else:
            # API call
            img_b64, mime = call_nano_banana(api_key, full_prompt)
            if img_b64:
                response_cache[cache_key] = (img_b64, mime)
        
        processing_time = time.time() - start_time
        
        if img_b64:
            background_tasks.add_task(log_usage, api_key, "generate", True, processing_time)
            return ImageResponse(
                image=img_b64,
                mime=mime,
                request_id=request_id,
                processing_time=processing_time
            )
        else:
            background_tasks.add_task(log_usage, api_key, "generate", False, processing_time)
            raise HTTPException(500, f"Generation failed: {mime}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}")
        background_tasks.add_task(log_usage, api_key, "generate", False, time.time() - start_time)
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/edit", response_model=ImageResponse)
async def edit_image(
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    prompt: str = Form(...),
    file: UploadFile = File(...),
    preserve_original: bool = Form(True)
):
    """Enhanced image editing with preservation control"""
    start_time = time.time()
    request_id = generate_request_id()
    
    try:
        if not check_rate_limit(api_key):
            raise HTTPException(429, "Rate limit exceeded")
        
        img_data, mime = b64encode_file(file)
        
        # Enhanced prompt with preservation control
        prompt_config = PromptConfig(preserve_original=preserve_original)
        enhanced_prompts = prompt_manager.get_enhanced_prompt("edit_image", prompt, prompt_config)
        system_prompt = random.choice(enhanced_prompts)
        full_prompt = f"{system_prompt} EDIT INSTRUCTIONS: {prompt}"
        
        img_b64, out_mime = call_nano_banana(api_key, full_prompt, images=[{'data': img_data, 'mime': mime}])
        processing_time = time.time() - start_time
        
        if img_b64:
            background_tasks.add_task(log_usage, api_key, "edit", True, processing_time)
            return ImageResponse(
                image=img_b64,
                mime=out_mime,
                request_id=request_id,
                processing_time=processing_time
            )
        else:
            background_tasks.add_task(log_usage, api_key, "edit", False, processing_time)
            raise HTTPException(500, f"Edit failed: {out_mime}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in edit_image: {str(e)}")
        background_tasks.add_task(log_usage, api_key, "edit", False, time.time() - start_time)
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/virtual_try_on", response_model=ImageResponse)
async def virtual_try_on(
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    product: UploadFile = File(...),
    person: UploadFile = File(...),
    prompt: str = Form("")
):
    """Enhanced virtual try-on with better prompt handling"""
    start_time = time.time()
    request_id = generate_request_id()
    
    try:
        if not check_rate_limit(api_key):
            raise HTTPException(429, "Rate limit exceeded")
        
        images = []
        for f in [product, person]:
            data, mime = b64encode_file(f)
            images.append({'data': data, 'mime': mime})
        
        # Enhanced prompt handling
        enhanced_prompts = prompt_manager.get_enhanced_prompt("virtual_try_on", prompt)
        system_prompt = random.choice(enhanced_prompts)
        full_prompt = system_prompt
        if prompt:
            full_prompt += f" ADDITIONAL INSTRUCTIONS: {prompt}"
        
        img_b64, out_mime = call_nano_banana(api_key, full_prompt, images=images)
        processing_time = time.time() - start_time
        
        if img_b64:
            background_tasks.add_task(log_usage, api_key, "virtual_try_on", True, processing_time)
            return ImageResponse(
                image=img_b64,
                mime=out_mime,
                request_id=request_id,
                processing_time=processing_time
            )
        else:
            background_tasks.add_task(log_usage, api_key, "virtual_try_on", False, processing_time)
            raise HTTPException(500, f"Virtual try-on failed: {out_mime}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in virtual_try_on: {str(e)}")
        background_tasks.add_task(log_usage, api_key, "virtual_try_on", False, time.time() - start_time)
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/create_ads", response_model=BatchResponse)
async def create_ads(
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    model: UploadFile = File(...),
    product: UploadFile = File(...),
    prompt: str = Form(""),
    variations: int = Form(3),
    style: str = Form("modern")
):
    """Enhanced ad creation with style control and better variation handling"""
    start_time = time.time()
    request_id = generate_request_id()
    
    try:
        if not check_rate_limit(api_key):
            raise HTTPException(429, "Rate limit exceeded")
        
        # Validate variations
        target = min(max(1, variations), config.MAX_AD_VARIATIONS)
        
        images = []
        for f in [model, product]:
            data, mime = b64encode_file(f)
            images.append({'data': data, 'mime': mime})
        
        results = []
        enhanced_prompts = prompt_manager.get_enhanced_prompt(
            "create_ads", 
            prompt, 
            PromptConfig(style=StylePreset(style))
        )
        system_prompt = enhanced_prompts[0]
        
        # Enhanced variation prompts
        variation_prompts = [
            "Create a lifestyle-focused ad with natural lighting and authentic setting",
            "Generate a high-fashion editorial style with dramatic lighting",
            "Produce a clean product-focused ad with minimalist composition",
            "Create a vibrant social media ad with bold colors and dynamic composition",
            "Generate a luxury brand aesthetic with elegant lighting and premium feel",
            "Produce a contemporary ad with modern aesthetics and sharp details"
        ]
        
        for i in range(target):
            variation_hint = variation_prompts[i % len(variation_prompts)]
            full_prompt = f"{system_prompt} {variation_hint}. AD VARIATION {i+1}/{target}."
            if prompt:
                full_prompt += f" USER BRIEF: {prompt}"
            
            img_b64, out_mime = call_nano_banana(api_key, full_prompt, images=images)
            if img_b64:
                results.append(ImageResponse(
                    image=img_b64,
                    mime=out_mime,
                    request_id=f"{request_id}_{i}",
                    processing_time=0  # Will be updated below
                ))
        
        processing_time = time.time() - start_time
        
        # Update processing times
        for result in results:
            result.processing_time = processing_time / len(results) if results else 0
        
        background_tasks.add_task(log_usage, api_key, "create_ads", bool(results), processing_time)
        return BatchResponse(
            results=results,
            total_generated=len(results),
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in create_ads: {str(e)}")
        background_tasks.add_task(log_usage, api_key, "create_ads", False, time.time() - start_time)
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/merge_images", response_model=ImageResponse)
async def merge_images(
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    files: List[UploadFile] = File(...),
    prompt: str = Form("")
):
    """Enhanced image merging with better multi-image handling"""
    start_time = time.time()
    request_id = generate_request_id()
    
    try:
        if not check_rate_limit(api_key):
            raise HTTPException(429, "Rate limit exceeded")
        
        if len(files) < 2:
            raise HTTPException(400, "At least 2 images required for merging")
        
        images = []
        for f in files[:5]:  # Limit to 5 images
            data, mime = b64encode_file(f)
            images.append({'data': data, 'mime': mime})
        
        enhanced_prompts = prompt_manager.get_enhanced_prompt("merge_images", prompt)
        system_prompt = random.choice(enhanced_prompts)
        full_prompt = system_prompt
        if prompt:
            full_prompt += f" MERGE INSTRUCTIONS: {prompt}"
        
        img_b64, out_mime = call_nano_banana(api_key, full_prompt, images=images)
        processing_time = time.time() - start_time
        
        if img_b64:
            background_tasks.add_task(log_usage, api_key, "merge_images", True, processing_time)
            return ImageResponse(
                image=img_b64,
                mime=out_mime,
                request_id=request_id,
                processing_time=processing_time
            )
        else:
            background_tasks.add_task(log_usage, api_key, "merge_images", False, processing_time)
            raise HTTPException(500, f"Merge failed: {out_mime}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in merge_images: {str(e)}")
        background_tasks.add_task(log_usage, api_key, "merge_images", False, time.time() - start_time)
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/generate_scenes", response_model=BatchResponse)
async def generate_scenes(
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    scene: UploadFile = File(...),
    prompt: str = Form(""),
    variations: int = Form(4)
):
    """Enhanced scene generation with better variation control"""
    start_time = time.time()
    request_id = generate_request_id()
    
    try:
        if not check_rate_limit(api_key):
            raise HTTPException(429, "Rate limit exceeded")
        
        target = min(max(1, variations), config.MAX_SCENE_VARIATIONS)
        data, mime = b64encode_file(scene)
        
        results = []
        enhanced_prompts = prompt_manager.get_enhanced_prompt("generate_scenes", prompt)
        system_prompt = enhanced_prompts[0]
        
        # Enhanced scene variation prompts
        scene_variations = [
            "Create a dawn version with soft morning light and misty atmosphere",
            "Generate a midday variation with bright, clear lighting and vibrant colors",
            "Produce a sunset scene with warm golden hour lighting and long shadows",
            "Create a night version with dramatic moonlight and artificial lighting",
            "Generate a rainy atmosphere with wet surfaces and moody lighting",
            "Produce a snowy transformation with winter lighting and frost details",
            "Create a foggy version with atmospheric perspective and soft edges",
            "Generate a stylized painterly interpretation with artistic flair"
        ]
        
        for i in range(target):
            variation_hint = scene_variations[i % len(scene_variations)]
            full_prompt = f"{system_prompt} {variation_hint}. SCENE VARIATION {i+1}/{target}."
            if prompt:
                full_prompt += f" USER DIRECTION: {prompt}"
            
            img_b64, out_mime = call_nano_banana(api_key, full_prompt, images=[{'data': data, 'mime': mime}])
            if img_b64:
                results.append(ImageResponse(
                    image=img_b64,
                    mime=out_mime,
                    request_id=f"{request_id}_{i}",
                    processing_time=0  # Will be updated below
                ))
        
        processing_time = time.time() - start_time
        
        # Update processing times
        for result in results:
            result.processing_time = processing_time / len(results) if results else 0
        
        background_tasks.add_task(log_usage, api_key, "generate_scenes", bool(results), processing_time)
        return BatchResponse(
            results=results,
            total_generated=len(results),
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_scenes: {str(e)}")
        background_tasks.add_task(log_usage, api_key, "generate_scenes", False, time.time() - start_time)
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/restore_old_image", response_model=ImageResponse)
async def restore_old_image(
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    file: UploadFile = File(...),
    prompt: str = Form(""),
    enhancement_level: str = Form("medium")
):
    """Enhanced image restoration with level control"""
    start_time = time.time()
    request_id = generate_request_id()
    
    try:
        if not check_rate_limit(api_key):
            raise HTTPException(429, "Rate limit exceeded")
        
        img_data, mime = b64encode_file(file)
        
        # Enhanced restoration with level control
        prompt_config = PromptConfig()
        enhanced_prompts = prompt_manager.get_enhanced_prompt("restore_old_image", prompt, prompt_config)
        system_prompt = random.choice(enhanced_prompts)
        full_prompt = f"{system_prompt} ENHANCEMENT LEVEL: {enhancement_level.upper()}."
        if prompt:
            full_prompt += f" SPECIFIC REQUESTS: {prompt}"
        
        img_b64, out_mime = call_nano_banana(api_key, full_prompt, images=[{'data': img_data, 'mime': mime}])
        processing_time = time.time() - start_time
        
        if img_b64:
            background_tasks.add_task(log_usage, api_key, "restore_old_image", True, processing_time)
            return ImageResponse(
                image=img_b64,
                mime=out_mime,
                request_id=request_id,
                processing_time=processing_time
            )
        else:
            background_tasks.add_task(log_usage, api_key, "restore_old_image", False, processing_time)
            raise HTTPException(500, f"Restoration failed: {out_mime}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in restore_old_image: {str(e)}")
        background_tasks.add_task(log_usage, api_key, "restore_old_image", False, time.time() - start_time)
        raise HTTPException(500, f"Internal server error: {str(e)}")

# === Additional Utility Endpoints ===
@app.post("/validate_prompt")
async def validate_prompt(prompt: str = Form(...), mode: str = Form("generate_image")):
    """Validate and analyze user prompts"""
    try:
        analysis = prompt_manager.validate_user_prompt(prompt, mode)
        return {
            "valid": analysis["is_valid"],
            "score": analysis["score"],
            "suggestions": analysis["suggestions"],
            "warnings": analysis["warnings"]
        }
    except Exception as e:
        raise HTTPException(500, f"Validation error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    return {
        "cache_usage": len(response_cache),
        "rate_limited_keys": len(rate_limit_cache),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config=None
    )