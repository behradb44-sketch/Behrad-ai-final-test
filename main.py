import os
import logging
from typing import Any

import httpx
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

APP_NAME = "هوش مصنوعی بهراد ایمیج"
APP_VERSION = "3.0.0"

ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID",
    ""
).strip()

API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN",
    ""
).strip()

IMAGE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
).strip()

TRANSLATION_MODEL = os.getenv(
    "CLOUDFLARE_TRANSLATION_MODEL",
    "@cf/meta/m2m100-1.2b"
).strip()

CHAT_MODEL = os.getenv(
    "CLOUDFLARE_CHAT_MODEL",
    "@cf/meta/llama-3.1-8b-instruct-fast"
).strip()

PORT = int(
    os.getenv("PORT", "8000")
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
    )
)

logger = logging.getLogger(
    "behrad-ai"
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
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
# CLOUDFLARE
# =========================================================

CF_BASE_URL = (
    "https://api.cloudflare.com/client/v4/"
    f"accounts/{ACCOUNT_ID}/ai/run"
)


def cf_url(model: str) -> str:
    return f"{CF_BASE_URL}/{model}"


def cf_headers() -> dict[str, str]:

    return {
        "Authorization":
            f"Bearer {API_TOKEN}",

        "Content-Type":
            "application/json"
    }


def cloudflare_ready() -> bool:

    return bool(
        ACCOUNT_ID and
        API_TOKEN
    )


# =========================================================
# REQUEST MODELS
# =========================================================

class FrontendMessage(BaseModel):

    role: str

    content: Any

    image: Any | None = None

    sources: list[Any] = Field(
        default_factory=list
    )

    chart: Any | None = None


class ChatRequest(BaseModel):

    # ساختار فعلی frontend
    messages: list[FrontendMessage] = Field(
        default_factory=list
    )

    # سازگاری با نسخه قدیمی
    message: str | None = None


class ImageRequest(BaseModel):

    prompt: str = Field(
        min_length=1,
        max_length=2048
    )

    steps: int = Field(
        default=4,
        ge=1,
        le=8
    )


class TranslateRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=4096
    )


# =========================================================
# RESPONSE HELPERS
# =========================================================

def cloudflare_error(
    response: httpx.Response
) -> str:

    try:
        data = response.json()

    except Exception:

        return (
            response.text[:3000]
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

        result = []

        for error in errors:

            if isinstance(
                error,
                dict
            ):

                message = error.get(
                    "message"
                )

                if message:
                    result.append(
                        str(message)
                    )

            else:

                result.append(
                    str(error)
                )

        if result:
            return " | ".join(result)

    return str(data)[:3000]


def get_cf_result(
    data: dict
) -> Any:

    if not data.get(
        "success",
        False
    ):

        raise RuntimeError(
            str(
                data.get(
                    "errors",
                    "Cloudflare request failed."
                )
            )
        )

    return data.get(
        "result"
    )


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_farsi(
    text: str
) -> str:

    replacements = {

        "ي": "ی",

        "ى": "ی",

        "ك": "ک",

        "ة": "ه",

        "ۀ": "ه",

        "\u200c": " "

    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    return " ".join(
        text.split()
    ).strip()


# =========================================================
# TRANSLATION
# =========================================================

async def translate_to_english(
    text: str
) -> str:

    """
    Persian -> English

    با مدل M2M100.

    اگر ترجمه شکست بخورد،
    خود متن اصلی برگردانده می‌شود تا
    ساخت تصویر کاملاً از کار نیفتد.
    """

    text = normalize_farsi(
        text
    )

    if not text:
        return text

    # اگر متن کاملاً انگلیسی باشد،
    # نیازی به ترجمه نیست.
    has_persian = any(
        "\u0600" <= char <= "\u06ff"
        for char in text
    )

    if not has_persian:
        return text

    payload = {

        "text": text,

        "source_lang": "fa",

        "target_lang": "en"

    }

    timeout = httpx.Timeout(
        connect=20,
        read=120,
        write=30,
        pool=20
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        response = await client.post(

            cf_url(
                TRANSLATION_MODEL
            ),

            headers=cf_headers(),

            json=payload
        )

    if response.status_code >= 400:

        logger.warning(
            "Translation failed: %s",
            cloudflare_error(response)
        )

        return text

    try:

        data = response.json()

    except Exception:

        return text

    try:

        result = get_cf_result(
            data
        )

    except Exception:

        return text

    if isinstance(
        result,
        dict
    ):

        # ساختارهای متداول M2M100
        for key in (
            "translated_text",
            "translation",
            "text",
            "response"
        ):

            value = result.get(
                key
            )

            if isinstance(
                value,
                str
            ) and value.strip():

                return value.strip()

    if isinstance(
        result,
        str
    ):

        return result.strip()

    return text


# =========================================================
# IMAGE GENERATION
# =========================================================

async def generate_image(
    prompt: str,
    steps: int = 4
) -> str:

    if not cloudflare_ready():

        raise RuntimeError(
            "Cloudflare is not configured."
        )

    prompt = prompt.strip()

    if not prompt:

        raise RuntimeError(
            "Image prompt is empty."
        )

    # محدودیت مدل
    steps = max(
        1,
        min(
            int(steps),
            8
        )
    )

    # -----------------------------------------------------
    # FLUX.1 SCHNELL
    # -----------------------------------------------------

    payload = {

        "prompt": prompt,

        "steps": steps

    }

    timeout = httpx.Timeout(
        connect=20,
        read=180,
        write=30,
        pool=20
    )

    logger.info(
        "Generating image..."
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        try:

            response = await client.post(

                cf_url(
                    IMAGE_MODEL
                ),

                headers=cf_headers(),

                json=payload
            )

        except httpx.TimeoutException:

            raise RuntimeError(
                "Image generation timed out."
            )

        except httpx.RequestError as exc:

            raise RuntimeError(
                f"Cloudflare connection error: {exc}"
            )

    if response.status_code >= 400:

        raise RuntimeError(
            "Cloudflare Image Error: "
            + cloudflare_error(response)
        )

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            "Cloudflare returned invalid JSON."
        )

    result = get_cf_result(
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
        "data:image/jpeg;base64,"
        + image
    )


# =========================================================
# IMAGE DETECTION
# =========================================================

IMAGE_PHRASES = [

    "عکس بساز",

    "تصویر بساز",

    "عکس درست کن",

    "تصویر درست کن",

    "عکس ایجاد کن",

    "تصویر ایجاد کن",

    "عکس تولید کن",

    "تصویر تولید کن",

    "عکس بکش",

    "تصویر بکش",

    "بساز عکس",

    "بساز تصویر",

    "generate image",

    "create image",

    "create a picture",

    "make an image",

    "make a picture",

    "draw me",

    "draw a"

]


def is_image_request(
    text: str
) -> bool:

    value = text.lower().strip()

    return any(
        phrase in value
        for phrase in IMAGE_PHRASES
    )


def extract_image_prompt(
    text: str
) -> str:

    result = text.strip()

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

        "عکس بکش",

        "تصویر بکش"

    ]

    lower = result.lower()

    for prefix in prefixes:

        if lower.startswith(
            prefix.lower()
        ):

            result = result[
                len(prefix):
            ].strip()

            break

    return result or text


# =========================================================
# CHAT
# =========================================================

SYSTEM_PROMPT = """
تو BEHRAD AI هستی.

با کاربر طبیعی، دوستانه و راحت صحبت کن.

اگر کاربر فارسی حرف می‌زند، فارسی جواب بده.

لحن فارسی می‌تواند عامیانه و دوستانه باشد.

از Markdown استفاده کن.

وقتی لازم است:
- عنوان بساز
- متن را بولد کن
- لیست بساز
- جدول بساز
- کد را داخل code block قرار بده

پاسخ‌ها را واضح و کاربردی بده.

اگر چیزی را نمی‌دانی، وانمود نکن که می‌دانی.

ادعا نکن که در وب جستجو کرده‌ای مگر اینکه
واقعاً ابزار جستجو در اختیار تو باشد.

اگر کاربر درخواست ساخت تصویر کرد،
سیستم برنامه می‌تواند آن را به ابزار ساخت
تصویر ارسال کند.
""".strip()


def frontend_messages_to_cf(
    messages: list[FrontendMessage]
) -> list[dict[str, str]]:

    result = []

    for item in messages:

        role = (
            item.role
            .strip()
            .lower()
        )

        if role not in {
            "system",
            "user",
            "assistant"
        }:

            continue

        content = item.content

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

                    value = part.get(
                        "text"
                    )

                    if value:
                        parts.append(
                            str(value)
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

        result.append({

            "role": role,

            "content": content

        })

    return result


async def chat(
    messages: list[dict[str, str]]
) -> str:

    if not cloudflare_ready():

        raise RuntimeError(
            "Cloudflare is not configured."
        )

    # تاریخچه بیش از حد بزرگ نشود
    messages = messages[-30:]

    cf_messages = [

        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT
        }

    ]

    cf_messages.extend(
        messages
    )

    payload = {

        "messages":
            cf_messages,

        "max_tokens":
            1200,

        "temperature":
            0.7
    }

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

                cf_url(
                    CHAT_MODEL
                ),

                headers=cf_headers(),

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
            + cloudflare_error(response)
        )

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            "Cloudflare returned invalid JSON."
        )

    result = get_cf_result(
        data
    )

    if isinstance(
        result,
        dict
    ):

        answer = result.get(
            "response"
        )

        if isinstance(
            answer,
            str
        ):

            return answer.strip()

        for key in (
            "text",
            "output",
            "generated_text"
        ):

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
        "Cloudflare returned an empty AI response."
    )


# =========================================================
# HOME
# =========================================================

@app.get("/")
async def home():

    index_file = os.path.join(
        STATIC_DIR,
        "index.html"
    )

    if os.path.isfile(
        index_file
    ):

        return FileResponse(
            index_file
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
            "هوش مصنوعی کارگران کلودفلر",

        "image_model":
            IMAGE_MODEL,

        "translation_model":
            TRANSLATION_MODEL,

        "chat_model":
            CHAT_MODEL,

        "cloudflare_configured":
            cloudflare_ready()
    }


# =========================================================
# IMAGE ENDPOINT
# =========================================================

@app.post(
    "/api/generate-image"
)
async def api_generate_image(
    request: ImageRequest
):

    try:

        # ترجمه فارسی → انگلیسی
        english_prompt = (
            await translate_to_english(
                request.prompt
            )
        )

        logger.info(
            "Image prompt translated."
        )

        image = await generate_image(

            prompt=english_prompt,

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
                request.prompt,

            "translated_prompt":
                english_prompt

        }

    except RuntimeError as exc:

        logger.error(
            "Image generation error: %s",
            exc
        )

        raise HTTPException(

            status_code=502,

            detail=str(exc)
        )

    except Exception:

        logger.exception(
            "Unexpected image error."
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Unexpected image generation error."
            )
        )


# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post(
    "/api/chat"
)
async def api_chat(
    request: ChatRequest
):

    messages = (
        frontend_messages_to_cf(
            request.messages
        )
    )

    # سازگاری با نسخه قدیمی
    if request.message:

        text = request.message.strip()

        if text:

            messages.append({

                "role":
                    "user",

                "content":
                    text
            })

    if not messages:

        raise HTTPException(

            status_code=400,

            detail=(
                "No messages were provided."
            )
        )

    # آخرین پیام کاربر
    latest_user_message = None

    for item in reversed(
        messages
    ):

        if item["role"] == "user":

            latest_user_message = (
                item["content"]
            )

            break

    # =====================================================
    # IMAGE REQUEST
    # =====================================================

    if (
        latest_user_message
        and
        is_image_request(
            latest_user_message
        )
    ):

        try:

            image_prompt = (
                extract_image_prompt(
                    latest_user_message
                )
            )

            english_prompt = (
                await translate_to_english(
                    image_prompt
                )
            )

            image = await generate_image(

                prompt=english_prompt,

                steps=4
            )

            return {

                "success":
                    True,

                "type":
                    "image",

                "message":
                    "تصویرت آماده شد 😎🔥",

                "image":
                    image,

                "prompt":
                    image_prompt,

                "translated_prompt":
                    english_prompt,

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
            "Unexpected chat error."
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Unexpected AI error."
            )
        )


# =========================================================
# TRANSLATION ENDPOINT
# =========================================================

@app.post(
    "/api/translate"
)
async def api_translate(
    request: TranslateRequest
):

    try:

        translated = (
            await translate_to_english(
                request.text
            )
        )

        return {

            "success":
                True,

            "source":
                request.text,

            "translation":
                translated

        }

    except Exception as exc:

        logger.exception(
            "Translation error."
        )

        raise HTTPException(

            status_code=502,

            detail=str(exc)
        )


# =========================================================
# 404
# =========================================================

@app.exception_handler(404)
async def not_found(
    request,
    exc
):

    return {
        "success":
            False,

        "error":
            "Not found"
    }


# =========================================================
# LOCAL RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=PORT,

        reload=False
            )
