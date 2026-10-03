import os
import re
import base64
import logging
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

APP_NAME = "BEHRAD AI"
APP_VERSION = "2.1.0"

CLOUDFLARE_ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID",
    ""
).strip()

CLOUDFLARE_API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN",
    ""
).strip()

CLOUDFLARE_CHAT_MODEL = os.getenv(
    "CLOUDFLARE_CHAT_MODEL",
    "@cf/meta/llama-3.1-8b-instruct"
).strip()

CLOUDFLARE_IMAGE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
).strip()

PORT = int(os.getenv("PORT", "8000"))

BASE_CF_URL = (
    "https://api.cloudflare.com/client/v4/accounts/"
    f"{CLOUDFLARE_ACCOUNT_ID}/ai/run"
)

CF_CHAT_URL = (
    f"{BASE_CF_URL}/{CLOUDFLARE_CHAT_MODEL}"
)

CF_IMAGE_URL = (
    f"{BASE_CF_URL}/{CLOUDFLARE_IMAGE_MODEL}"
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger("behrad-ai")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# STATIC FILES
# =========================================================

STATIC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static",
)

if os.path.isdir(STATIC_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )


# =========================================================
# MODELS
# =========================================================

class ChatMessage(BaseModel):
    role: str = Field(
        default="user",
        description="user / assistant / system",
    )
    content: str = Field(
        min_length=1,
        max_length=20000,
    )


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=20000,
    )

    history: list[ChatMessage] = Field(
        default_factory=list,
    )


class ImageRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=2048,
    )

    steps: int = Field(
        default=4,
        ge=1,
        le=8,
    )


# =========================================================
# HELPERS
# =========================================================

def cloudflare_configured() -> bool:
    return bool(
        CLOUDFLARE_ACCOUNT_ID
        and CLOUDFLARE_API_TOKEN
    )


def cloudflare_headers() -> dict[str, str]:
    return {
        "Authorization": (
            f"Bearer {CLOUDFLARE_API_TOKEN}"
        ),
        "Content-Type": "application/json",
    }


def normalize_text(text: str) -> str:
    text = text.strip()

    # Arabic/Persian normalization
    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "\u200c": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def looks_like_image_request(text: str) -> bool:
    """
    تشخیص ساده درخواست تصویر.

    این تابع قرار نیست جای مدل زبانی را بگیرد؛
    فقط برای تشخیص سریع درخواست‌های واضح تصویر است.
    """

    text = normalize_text(text).lower()

    image_words = [
        "بساز",
        "ساخت",
        "ایجاد کن",
        "تولید کن",
        "تصویر",
        "عکس",
        "عکسی",
        "عکس بساز",
        "تصویر بساز",
        "تصویر ایجاد کن",
        "تصویر تولید کن",
        "generate image",
        "generate a picture",
        "create image",
        "create a picture",
        "make an image",
        "make a picture",
        "draw",
        "draw me",
    ]

    return any(
        word in text
        for word in image_words
    )


def clean_image_prompt(text: str) -> str:
    """
    درخواست فارسی کاربر را برای موتور تصویر
    به یک prompt تمیز تبدیل می‌کند.

    در این نسخه ترجمه به‌صورت جداگانه انجام نمی‌شود.
    اگر frontend/backend شما قبلاً مترجم دارد،
    می‌تواند prompt انگلیسی را مستقیماً بفرستد.
    """

    text = normalize_text(text)

    patterns = [
        r"^(?:یک|یه)?\s*عکس\s*(?:از)?\s*",
        r"^(?:یک|یه)?\s*تصویر\s*(?:از)?\s*",
        r"^(?:عکس|تصویر)\s*بساز\s*(?:از)?\s*",
        r"^(?:عکس|تصویر)\s*ایجاد\s*کن\s*(?:از)?\s*",
        r"^(?:عکس|تصویر)\s*تولید\s*کن\s*(?:از)?\s*",
    ]

    cleaned = text

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    cleaned = cleaned.strip()

    return cleaned or text


def extract_cloudflare_error(
    response: httpx.Response,
) -> str:

    try:
        data = response.json()
    except Exception:
        text = response.text.strip()

        if text:
            return text[:4000]

        return (
            f"Cloudflare returned HTTP "
            f"{response.status_code}"
        )

    errors = data.get("errors")

    if isinstance(errors, list):
        messages = []

        for error in errors:
            if isinstance(error, dict):
                message = error.get("message")

                if message:
                    messages.append(
                        str(message)
                    )
            else:
                messages.append(str(error))

        if messages:
            return " | ".join(messages)

    return str(data)[:4000]


def extract_result(data: dict[str, Any]) -> Any:
    if not data.get("success", False):
        errors = data.get("errors", [])

        raise RuntimeError(
            str(errors)
        )

    return data.get("result")


# =========================================================
# CLOUDFLARE IMAGE
# =========================================================

async def generate_image(
    prompt: str,
    steps: int = 4,
) -> str:

    if not cloudflare_configured():
        raise RuntimeError(
            "Cloudflare is not configured."
        )

    prompt = prompt.strip()

    if not prompt:
        raise RuntimeError(
            "Image prompt is empty."
        )

    steps = max(
        1,
        min(
            8,
            int(steps),
        ),
    )

    # =====================================================
    # IMPORTANT:
    #
    # FLUX.1 Schnell currently accepts:
    #   prompt
    #   steps
    #
    # DO NOT send seed / width / height / guidance /
    # negative_prompt to this model.
    # =====================================================

    payload = {
        "prompt": prompt,
        "steps": steps,
    }

    logger.info(
        "Generating image with model=%s steps=%s",
        CLOUDFLARE_IMAGE_MODEL,
        steps,
    )

    timeout = httpx.Timeout(
        connect=20.0,
        read=180.0,
        write=30.0,
        pool=20.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        try:
            response = await client.post(
                CF_IMAGE_URL,
                headers=cloudflare_headers(),
                json=payload,
            )

        except httpx.TimeoutException:
            raise RuntimeError(
                "Cloudflare image request timed out."
            )

        except httpx.RequestError as exc:
            logger.exception(
                "Cloudflare connection error"
            )

            raise RuntimeError(
                f"Cloudflare connection error: {exc}"
            )

    if response.status_code >= 400:
        error = extract_cloudflare_error(
            response
        )

        logger.error(
            "Cloudflare image error: %s",
            error,
        )

        raise RuntimeError(
            f"Cloudflare Image Error: {error}"
        )

    try:
        data = response.json()

    except Exception:
        raise RuntimeError(
            "Cloudflare returned invalid JSON."
        )

    result = extract_result(data)

    if not isinstance(result, dict):
        raise RuntimeError(
            "Cloudflare returned an unexpected image result."
        )

    image_base64 = result.get("image")

    if not image_base64:
        raise RuntimeError(
            "Cloudflare returned no image data."
        )

    if not isinstance(
        image_base64,
        str,
    ):
        raise RuntimeError(
            "Cloudflare image data is invalid."
        )

    # Remove accidental data URI prefix if provider
    # ever returns one.
    if image_base64.startswith(
        "data:image"
    ):
        return image_base64

    return (
        "data:image/jpeg;base64,"
        + image_base64
    )


# =========================================================
# CLOUDFLARE CHAT
# =========================================================

SYSTEM_PROMPT = """
You are BEHRAD AI.

You are a helpful general-purpose AI assistant.

Rules:
- Answer naturally and clearly.
- If the user writes Persian, answer Persian.
- If the user writes English, answer English.
- Do not mention internal APIs, Cloudflare, models,
  providers, tokens, prompts, or implementation details
  unless the user explicitly asks about the technology.
- Use Markdown when useful.
- You may use headings, bullet lists, numbered lists,
  tables and code blocks when appropriate.
- Do not create fake citations or fake web results.
- Do not claim that you searched the web unless a real
  web-search tool was actually used.
- If the user asks to create an image, the application
  may handle that request separately.
""".strip()


async def chat_with_cloudflare(
    message: str,
    history: list[ChatMessage],
) -> str:

    if not cloudflare_configured():
        raise RuntimeError(
            "Cloudflare is not configured."
        )

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # Keep history under a reasonable limit.
    safe_history = history[-20:]

    for item in safe_history:

        role = item.role.strip().lower()

        if role not in {
            "user",
            "assistant",
            "system",
        }:
            role = "user"

        content = item.content.strip()

        if not content:
            continue

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": message.strip(),
        }
    )

    # Cloudflare's model-specific REST API accepts the
    # model input directly. For chat-capable models,
    # prompt/messages support depends on the selected model.
    #
    # The default below is compatible with the classic
    # prompt-based instruct model.
    combined_prompt_parts = []

    for item in messages:
        combined_prompt_parts.append(
            f"{item['role'].upper()}: "
            f"{item['content']}"
        )

    combined_prompt = "\n\n".join(
        combined_prompt_parts
    )

    payload = {
        "prompt": combined_prompt,
        "max_tokens": 1200,
        "temperature": 0.7,
    }

    timeout = httpx.Timeout(
        connect=20.0,
        read=120.0,
        write=30.0,
        pool=20.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        try:
            response = await client.post(
                CF_CHAT_URL,
                headers=cloudflare_headers(),
                json=payload,
            )

        except httpx.TimeoutException:
            raise RuntimeError(
                "Cloudflare chat request timed out."
            )

        except httpx.RequestError as exc:
            logger.exception(
                "Cloudflare chat connection error"
            )

            raise RuntimeError(
                f"Cloudflare connection error: {exc}"
            )

    if response.status_code >= 400:

        error = extract_cloudflare_error(
            response
        )

        logger.error(
            "Cloudflare chat error: %s",
            error,
        )

        raise RuntimeError(
            f"Cloudflare Chat Error: {error}"
        )

    try:
        data = response.json()

    except Exception:
        raise RuntimeError(
            "Cloudflare returned invalid JSON."
        )

    result = extract_result(data)

    if isinstance(result, dict):

        answer = result.get(
            "response"
        )

        if isinstance(answer, str):
            return answer.strip()

        # Some models may return text instead.
        for key in (
            "text",
            "output",
            "generated_text",
        ):
            value = result.get(key)

            if isinstance(value, str):
                return value.strip()

    if isinstance(result, str):
        return result.strip()

    raise RuntimeError(
        "Cloudflare returned an unexpected chat result."
    )


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
async def home():

    index_file = os.path.join(
        STATIC_DIR,
        "index.html",
    )

    if os.path.isfile(index_file):
        return FileResponse(
            index_file
        )

    return JSONResponse(
        {
            "service": APP_NAME,
            "status": "ok",
            "message": (
                "BEHRAD AI backend is running."
            ),
        }
    )


@app.get("/api/health")
async def health():

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
        "provider": "Cloudflare Workers AI",
        "chat_model": CLOUDFLARE_CHAT_MODEL,
        "image_model": CLOUDFLARE_IMAGE_MODEL,
        "cloudflare_configured": (
            cloudflare_configured()
        ),
    }


@app.get("/api/test")
async def test():

    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
    }


# =========================================================
# IMAGE ENDPOINT
# =========================================================

@app.post("/api/generate-image")
async def api_generate_image(
    request: ImageRequest,
):

    try:

        image = await generate_image(
            prompt=request.prompt,
            steps=request.steps,
        )

        return {
            "success": True,
            "image": image,
            "prompt": request.prompt,
        }

    except RuntimeError as exc:

        logger.error(
            "Image generation failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    except Exception as exc:

        logger.exception(
            "Unexpected image generation error"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected image generation error."
            ),
        )


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post("/api/chat")
async def api_chat(
    request: ChatRequest,
):

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # -----------------------------------------------------
    # IMAGE REQUEST DETECTION
    # -----------------------------------------------------

    if looks_like_image_request(message):

        image_prompt = clean_image_prompt(
            message
        )

        try:

            image = await generate_image(
                prompt=image_prompt,
                steps=4,
            )

            return {
                "success": True,
                "type": "image",
                "message": (
                    "تصویرت آماده شد."
                ),
                "image": image,
                "prompt": image_prompt,
            }

        except RuntimeError as exc:

            logger.error(
                "Automatic image generation failed: %s",
                exc,
            )

            # We intentionally return a normal chat-style
            # error rather than crashing the application.
            return {
                "success": False,
                "type": "error",
                "message": str(exc),
            }

    # -----------------------------------------------------
    # NORMAL CHAT
    # -----------------------------------------------------

    try:

        answer = await chat_with_cloudflare(
            message=message,
            history=request.history,
        )

        return {
            "success": True,
            "type": "text",
            "message": answer,
        }

    except RuntimeError as exc:

        logger.error(
            "Chat failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    except Exception:

        logger.exception(
            "Unexpected chat error"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected AI error."
            ),
        )


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.exception_handler(404)
async def not_found_handler(
    request,
    exc,
):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Not found",
        },
    )


@app.exception_handler(500)
async def server_error_handler(
    request,
    exc,
):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
        },
    )


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
            )
