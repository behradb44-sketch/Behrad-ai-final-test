import os
import logging
import base64
from typing import Any

import httpx
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field


# =========================================================
# ENV
# =========================================================

load_dotenv()

APP_NAME = "BEHRAD AI"
APP_VERSION = "2.2.0"

ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID",
    ""
).strip()

API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN",
    ""
).strip()

CHAT_MODEL = os.getenv(
    "CLOUDFLARE_CHAT_MODEL",
    "@cf/meta/llama-3.1-8b-instruct"
).strip()

IMAGE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
).strip()

PORT = int(
    os.getenv("PORT", "8000")
)


# =========================================================
# CLOUDFLARE URL
# =========================================================

CLOUDFLARE_RUN_URL = (
    "https://api.cloudflare.com/client/v4/"
    f"accounts/{ACCOUNT_ID}/ai/run"
)


def model_url(model: str) -> str:
    return (
        f"{CLOUDFLARE_RUN_URL}/"
        f"{model}"
    )


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "behrad-ai"
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# STATIC
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)


if os.path.isdir(STATIC_DIR):

    app.mount(
        "/static",
        StaticFiles(
            directory=STATIC_DIR
        ),
        name="static"
    )


# =========================================================
# DATA MODELS
# =========================================================

class Message(BaseModel):

    role: str

    content: str | list[Any]

    image: Any | None = None

    sources: list[Any] = Field(
        default_factory=list
    )

    chart: Any | None = None


class ChatRequest(BaseModel):

    # فرانت‌اند فعلی
    messages: list[Message] = Field(
        default_factory=list
    )

    # برای سازگاری با API قبلی
    message: str | None = None

    # تنظیمات اختیاری
    stream: bool = False


class ImageRequest(BaseModel):

    prompt: str = Field(
        min_length=1,
        max_length=4096
    )

    steps: int = Field(
        default=4,
        ge=1,
        le=8
    )


# =========================================================
# HELPERS
# =========================================================

def is_configured() -> bool:

    return bool(
        ACCOUNT_ID
        and API_TOKEN
    )


def headers() -> dict[str, str]:

    return {
        "Authorization":
            f"Bearer {API_TOKEN}",

        "Content-Type":
            "application/json"
    }


def extract_error(
    response: httpx.Response
) -> str:

    try:

        data = response.json()

    except Exception:

        return (
            response.text[:4000]
            or
            f"HTTP {response.status_code}"
        )

    errors = data.get(
        "errors",
        []
    )

    if isinstance(
        errors,
        list
    ):

        messages = []

        for error in errors:

            if isinstance(
                error,
                dict
            ):

                msg = error.get(
                    "message"
                )

                if msg:
                    messages.append(
                        str(msg)
                    )

            else:

                messages.append(
                    str(error)
                )

        if messages:

            return " | ".join(
                messages
            )

    return str(data)[:4000]


def get_result(
    data: dict
) -> Any:

    if data.get(
        "success",
        False
    ):

        return data.get(
            "result"
        )

    raise RuntimeError(
        str(
            data.get(
                "errors",
                "Unknown Cloudflare error"
            )
        )
    )


# =========================================================
# IMAGE REQUEST DETECTION
# =========================================================

IMAGE_KEYWORDS = [

    "عکس بساز",

    "تصویر بساز",

    "عکس درست کن",

    "تصویر درست کن",

    "عکس ایجاد کن",

    "تصویر ایجاد کن",

    "عکس تولید کن",

    "تصویر تولید کن",

    "یک عکس بساز",

    "یه عکس بساز",

    "یک تصویر بساز",

    "یه تصویر بساز",

    "generate image",

    "generate a picture",

    "create image",

    "create a picture",

    "make an image",

    "make a picture",

    "draw me",

    "draw a"

]


def wants_image(
    text: str
) -> bool:

    value = text.strip().lower()

    return any(
        keyword in value
        for keyword in IMAGE_KEYWORDS
    )


def image_prompt_from_text(
    text: str
) -> str:

    value = text.strip()

    prefixes = [

        "یک عکس بساز",

        "یه عکس بساز",

        "عکس بساز",

        "یک تصویر بساز",

        "یه تصویر بساز",

        "تصویر بساز",

        "عکس درست کن",

        "تصویر درست کن",

        "عکس ایجاد کن",

        "تصویر ایجاد کن",

        "عکس تولید کن",

        "تصویر تولید کن",

        "generate image",

        "create image",

        "create a picture",

        "make an image"

    ]

    lower = value.lower()

    for prefix in prefixes:

        if lower.startswith(
            prefix.lower()
        ):

            value = value[
                len(prefix):
            ].strip()

            break

    return value or text


# =========================================================
# IMAGE GENERATION
# =========================================================

async def generate_image(
    prompt: str,
    steps: int = 4
) -> str:

    if not is_configured():

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
            int(steps),
            8
        )
    )

    # =====================================================
    # FLUX.1 SCHNELL
    #
    # مهم:
    # seed عمداً ارسال نمی‌شود.
    # =====================================================

    payload = {

        "prompt": prompt,

        "steps": steps

    }

    logger.info(
        "Generating image: %s",
        prompt
    )

    timeout = httpx.Timeout(
        connect=20,
        read=180,
        write=30,
        pool=20
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        try:

            response = await client.post(

                model_url(
                    IMAGE_MODEL
                ),

                headers=headers(),

                json=payload

            )

        except httpx.TimeoutException:

            raise RuntimeError(
                "Image generation timed out."
            )

        except httpx.RequestError as exc:

            raise RuntimeError(
                f"Connection error: {exc}"
            )

    if response.status_code >= 400:

        raise RuntimeError(
            "Cloudflare Image Error: "
            + extract_error(response)
        )

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            "Invalid Cloudflare response."
        )

    result = get_result(
        data
    )

    if not isinstance(
        result,
        dict
    ):

        raise RuntimeError(
            "Invalid image result."
        )

    image = result.get(
        "image"
    )

    if not image:

        raise RuntimeError(
            "Cloudflare returned no image."
        )

    if image.startswith(
        "data:image"
    ):

        return image

    return (
        "data:image/png;base64,"
        + image
    )


# =========================================================
# CHAT
# =========================================================

SYSTEM_PROMPT = """
تو BEHRAD AI هستی.

با کاربر طبیعی و دوستانه صحبت کن.

اگر کاربر فارسی صحبت می‌کند،
فارسی جواب بده.

اگر کاربر انگلیسی صحبت می‌کند،
انگلیسی جواب بده.

لحن فارسی می‌تواند دوستانه و عامیانه باشد،
ولی واضح و محترمانه بمان.

از Markdown استفاده کن.

وقتی لازم است از این قابلیت‌ها استفاده کن:

- عنوان
- بولت لیست
- لیست شماره‌دار
- جدول
- کد
- متن بولد

هیچ‌وقت درباره API، مدل داخلی،
Cloudflare یا پیاده‌سازی داخلی صحبت نکن،
مگر اینکه کاربر مستقیماً درباره آن بپرسد.

ادعا نکن که در وب جستجو کرده‌ای،
مگر اینکه واقعاً ابزار جستجو در اختیار تو باشد.

اگر کاربر از تو درخواست ساخت تصویر کرد،
سیستم برنامه ممکن است درخواست را به ابزار
ساخت تصویر ارسال کند.
""".strip()


def clean_messages(
    messages: list[Message]
) -> list[dict[str, str]]:

    cleaned = []

    for item in messages:

        role = item.role.lower().strip()

        if role not in {
            "system",
            "user",
            "assistant"
        }:

            continue

        content = item.content

        # اگر content لیست بود
        if isinstance(
            content,
            list
        ):

            parts = []

            for part in content:

                if isinstance(
                    part,
                    dict
                ):

                    text = part.get(
                        "text"
                    )

                    if text:
                        parts.append(
                            str(text)
                        )

            content = "\n".join(
                parts
            )

        if not isinstance(
            content,
            str
        ):

            content = str(
                content
            )

        content = content.strip()

        if not content:
            continue

        cleaned.append({

            "role": role,

            "content": content

        })

    return cleaned


async def chat(
    messages: list[dict[str, str]]
) -> str:

    if not is_configured():

        raise RuntimeError(
            "Cloudflare is not configured."
        )

    # -----------------------------------------------------
    # محدود کردن history برای جلوگیری از درخواست خیلی
    # بزرگ
    # -----------------------------------------------------

    messages = messages[-30:]

    final_messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }

    ]

    final_messages.extend(
        messages
    )

    # -----------------------------------------------------
    # Cloudflare Workers AI
    #
    # برای مدل‌های متنی مدرن Cloudflare،
    # messages به صورت رسمی پشتیبانی می‌شود.
    # -----------------------------------------------------

    payload = {

        "messages":
            final_messages,

        "max_tokens":
            1500,

        "temperature":
            0.7

    }

    logger.info(
        "Sending chat request (%d messages)",
        len(final_messages)
    )

    timeout = httpx.Timeout(
        connect=20,
        read=120,
        write=30,
        pool=20
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        try:

            response = await client.post(

                model_url(
                    CHAT_MODEL
                ),

                headers=headers(),

                json=payload

            )

        except httpx.TimeoutException:

            raise RuntimeError(
                "AI request timed out."
            )

        except httpx.RequestError as exc:

            raise RuntimeError(
                f"AI connection error: {exc}"
            )

    if response.status_code >= 400:

        raise RuntimeError(
            "Cloudflare Chat Error: "
            + extract_error(response)
        )

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            "Cloudflare returned invalid JSON."
        )

    result = get_result(
        data
    )

    if isinstance(
        result,
        dict
    ):

        response_text = result.get(
            "response"
        )

        if isinstance(
            response_text,
            str
        ):

            return response_text.strip()

        # بعضی مدل‌ها ممکن است text برگردانند

        for key in [
            "text",
            "output",
            "generated_text"
        ]:

            value = result.get(
                key
            )

            if isinstance(
                value,
                str
            ):

                return value.strip()

    if isinstance(
        result,
        str
    ):

        return result.strip()

    raise RuntimeError(
        "Unexpected AI response."
    )


# =========================================================
# HOME
# =========================================================

@app.get("/")
async def home():

    index_path = os.path.join(
        STATIC_DIR,
        "index.html"
    )

    if os.path.isfile(
        index_path
    ):

        return FileResponse(
            index_path
        )

    return {

        "service":
            APP_NAME,

        "version":
            APP_VERSION,

        "status":
            "ok"

    }


# =========================================================
# HEALTH
# =========================================================

@app.get(
    "/api/health"
)
async def health():

    return {

        "service":
            APP_NAME,

        "version":
            APP_VERSION,

        "status":
            "ok",

        "provider":
            "Cloudflare Workers AI",

        "chat_model":
            CHAT_MODEL,

        "image_model":
            IMAGE_MODEL,

        "cloudflare_configured":
            is_configured()

    }


# =========================================================
# IMAGE API
# =========================================================

@app.post(
    "/api/generate-image"
)
async def generate_image_api(
    request: ImageRequest
):

    try:

        image = await generate_image(

            prompt=request.prompt,

            steps=request.steps

        )

        return {

            "success":
                True,

            "type":
                "image",

            "image":
                image,

            "prompt":
                request.prompt

        }

    except RuntimeError as exc:

        logger.error(
            "Image error: %s",
            exc
        )

        raise HTTPException(

            status_code=502,

            detail=str(exc)

        )

    except Exception:

        logger.exception(
            "Unexpected image error"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Unexpected image generation error."
            )

        )


# =========================================================
# CHAT API
# =========================================================

@app.post(
    "/api/chat"
)
async def chat_api(
    request: ChatRequest
):

    # =====================================================
    # سازگاری با هر دو فرمت:
    #
    # جدید:
    # {
    #   "messages": [...]
    # }
    #
    # قدیمی:
    # {
    #   "message": "سلام"
    # }
    # =====================================================

    messages = clean_messages(
        request.messages
    )

    if request.message:

        messages.append({

            "role":
                "user",

            "content":
                request.message.strip()

        })

    if not messages:

        raise HTTPException(

            status_code=400,

            detail=(
                "No message was provided."
            )

        )

    # آخرین پیام واقعی کاربر
    user_message = None

    for item in reversed(
        messages
    ):

        if item["role"] == "user":

            user_message = item[
                "content"
            ]

            break

    # =====================================================
    # IMAGE REQUEST
    # =====================================================

    if (
        user_message
        and
        wants_image(
            user_message
        )
    ):

        image_prompt = (
            image_prompt_from_text(
                user_message
            )
        )

        try:

            image = await generate_image(

                prompt=image_prompt,

                steps=4

            )

            return {

                "success":
                    True,

                "type":
                    "image",

                "message":
                    "تصویرت آماده شد 😎",

                "image":
                    image,

                "prompt":
                    image_prompt,

                "sources":
                    [],

                "chart":
                    None

            }

        except RuntimeError as exc:

            logger.error(
                "Automatic image error: %s",
                exc
            )

            return {

                "success":
                    False,

                "type":
                    "error",

                "message":
                    str(exc),

                "image":
                    None,

                "sources":
                    [],

                "chart":
                    None

            }

    # =====================================================
    # NORMAL CHAT
    # =====================================================

    try:

        answer = await chat(
            messages
        )

        return {

            "success":
                True,

            "type":
                "text",

            "message":
                answer,

            "image":
                None,

            "sources":
                [],

            "chart":
                None

        }

    except RuntimeError as exc:

        logger.error(
            "Chat error: %s",
            exc
        )

        raise HTTPException(

            status_code=502,

            detail=str(exc)

        )

    except Exception:

        logger.exception(
            "Unexpected chat error"
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Unexpected AI error."
            )

        )


# =========================================================
# 404
# =========================================================

@app.exception_handler(404)
async def not_found(
    request,
    exc
):

    return JSONResponse(

        status_code=404,

        content={

            "success":
                False,

            "error":
                "Not found"

        }

    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=PORT,

        reload=False

            )
