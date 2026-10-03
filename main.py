import os
import json
import time
import uuid
import re
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

APP_NAME = "BEHRAD AI"
APP_VERSION = "1.1.0-test"

CLOUDFLARE_ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID",
    ""
).strip()

CLOUDFLARE_API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN",
    ""
).strip()

CHAT_MODEL = os.getenv(
    "CLOUDFLARE_CHAT_MODEL",
    "@cf/openai/gpt-oss-20b"
).strip()

IMAGE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
).strip()


# ============================================================
# CLOUDFLARE URLS
# ============================================================

CF_CHAT_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions"
)

CF_IMAGE_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/"
    f"{IMAGE_MODEL}"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION
)


# ============================================================
# RATE LIMIT
# ============================================================

RATE_LIMIT = 20
RATE_WINDOW = 60

rate_history = defaultdict(deque)


def check_rate_limit(ip: str) -> bool:
    now = time.monotonic()

    queue = rate_history[ip]

    while queue and now - queue[0] > RATE_WINDOW:
        queue.popleft()

    if len(queue) >= RATE_LIMIT:
        return False

    queue.append(now)

    return True


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are BEHRAD AI.

You are a Persian-first general AI assistant.

IMPORTANT BEHAVIOR:

1. Answer normally when the user asks a normal question.

2. If the user asks for current information, latest information,
   current prices, recent news, websites, current products,
   availability, or explicitly asks to search the web,
   use the web_search tool.

3. If the user asks to create, generate, draw, make or visualize
   an image, use the generate_image tool.

4. When using generate_image:
   - Convert the user's request into a detailed English image prompt.
   - Preserve all important details.
   - Do not merely explain how to make the image.
   - Actually call generate_image.

5. If the user asks for a chart and there is useful numerical data,
   use create_chart.

6. Markdown is allowed.

7. Use headings when useful.

8. Use tables when useful.

9. Use code blocks for programming code.

10. Never claim that you searched the web unless web_search
    actually executed.

11. Never claim that an image was generated unless
    generate_image actually executed successfully.

12. Never invent sources.

13. Answer naturally in Persian when the user speaks Persian.

14. Do not mention internal implementation details unless
    the user specifically asks.

15. Keep answers useful and natural.

16. For image generation, always create a detailed English prompt
    suitable for a modern text-to-image model.

17. Do not add parameters such as seed to generate_image.
    The image tool accepts only the prompt and steps.
"""


# ============================================================
# TOOLS
# ============================================================

TOOLS = [

    # --------------------------------------------------------
    # WEB SEARCH
    # --------------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "web_search",

            "description": (
                "Search the public web for current or changing "
                "information such as prices, news, products, "
                "availability, recent facts, or explicit web searches."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description": "The exact search query."
                    },

                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8
                    }

                },

                "required": [
                    "query"
                ]
            }
        }
    },


    # --------------------------------------------------------
    # IMAGE GENERATION
    # --------------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "generate_image",

            "description": (
                "Generate an image when the user asks to create, "
                "generate, draw, make, design or visualize an image."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "prompt": {
                        "type": "string",
                        "description": (
                            "A detailed English prompt describing "
                            "exactly what the generated image should contain."
                        )
                    },

                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "description": (
                            "Number of diffusion steps. "
                            "Use 4 by default."
                        )
                    }

                },

                "required": [
                    "prompt"
                ]
            }
        }
    },


    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "create_chart",

            "description": (
                "Create a chart when useful numerical data "
                "is available."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "title": {
                        "type": "string"
                    },

                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "bar",
                            "line",
                            "doughnut"
                        ]
                    },

                    "labels": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "values": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        }
                    },

                    "unit": {
                        "type": "string"
                    }
                },

                "required": [
                    "title",
                    "chart_type",
                    "labels",
                    "values"
                ]
            }
        }
    }
]


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    messages: list[dict] = Field(
        default_factory=list
    )


# ============================================================
# CLOUDFLARE HELPERS
# ============================================================

def cloudflare_configured() -> bool:

    return bool(
        CLOUDFLARE_ACCOUNT_ID
        and
        CLOUDFLARE_API_TOKEN
    )


def cloudflare_headers() -> dict:

    return {
        "Authorization":
            f"Bearer {CLOUDFLARE_API_TOKEN}",

        "Content-Type":
            "application/json"
    }


def extract_cloudflare_error(
    response: httpx.Response
) -> str:

    try:

        data = response.json()

    except Exception:

        text = response.text.strip()

        return (
            text[:3000]
            if text
            else f"HTTP {response.status_code}"
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

                message = error.get(
                    "message"
                )

                if message:
                    messages.append(
                        str(message)
                    )

            else:

                messages.append(
                    str(error)
                )

        if messages:
            return " | ".join(messages)

    return json.dumps(
        data,
        ensure_ascii=False
    )[:3000]


# ============================================================
# MESSAGE CLEANING
# ============================================================

def clean_messages(
    messages: list[dict]
) -> list[dict]:

    cleaned = []

    for message in messages[-30:]:

        role = message.get(
            "role"
        )

        content = message.get(
            "content"
        )

        if role not in {
            "user",
            "assistant",
            "tool"
        }:

            continue

        # ----------------------------------------
        # TOOL MESSAGE
        # ----------------------------------------

        if role == "tool":

            cleaned.append({

                "role":
                    "tool",

                "tool_call_id":
                    str(
                        message.get(
                            "tool_call_id",
                            ""
                        )
                    ),

                "content":
                    str(
                        content or ""
                    )[:16000]
            })

            continue

        # ----------------------------------------
        # NORMAL MESSAGE
        # ----------------------------------------

        if isinstance(
            content,
            str
        ) and content.strip():

            cleaned.append({

                "role":
                    role,

                "content":
                    content[:16000]
            })

    return cleaned


def make_sse(
    event: str,
    data: dict
) -> str:

    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


# ============================================================
# CLOUDFLARE CHAT
# ============================================================

async def cloudflare_chat(
    client: httpx.AsyncClient,
    messages: list[dict]
):

    payload = {

        "model":
            CHAT_MODEL,

        "messages": [

            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT
            }

        ] + messages,

        "tools":
            TOOLS,

        "tool_choice":
            "auto",

        "temperature":
            0.5,

        "max_tokens":
            1600,

        "stream":
            False
    }

    response = await client.post(

        CF_CHAT_URL,

        headers=
            cloudflare_headers(),

        json=
            payload
    )

    if response.status_code >= 400:

        raise RuntimeError(
            "Cloudflare Chat Error: "
            + extract_cloudflare_error(
                response
            )
        )

    try:

        return response.json()

    except Exception:

        raise RuntimeError(
            "Cloudflare Chat returned invalid JSON."
        )


# ============================================================
# WEB SEARCH
# ============================================================

async def web_search(
    client: httpx.AsyncClient,
    query: str,
    max_results: int = 6
):

    query = query.strip()

    if not query:
        return []

    search_url = (
        "https://html.duckduckgo.com/html/?q="
        + quote_plus(query)
    )

    response = await client.get(

        search_url,

        headers={
            "User-Agent":
                (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "Chrome/140 Safari/537.36"
                )
        },

        follow_redirects=True,

        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    results = []

    for item in soup.select(
        ".result"
    ):

        title_element = item.select_one(
            ".result__a"
        )

        snippet_element = item.select_one(
            ".result__snippet"
        )

        if not title_element:
            continue

        title = title_element.get_text(
            " ",
            strip=True
        )

        url = title_element.get(
            "href",
            ""
        )

        if not url:
            continue

        snippet = ""

        if snippet_element:

            snippet = snippet_element.get_text(
                " ",
                strip=True
            )

        domain = urlparse(
            url
        ).netloc.lower()

        domain = domain.removeprefix(
            "www."
        )

        favicon = (
            "https://www.google.com/s2/favicons"
            f"?domain={quote_plus(domain)}&sz=64"
        )

        results.append({

            "title":
                title,

            "url":
                url,

            "domain":
                domain,

            "snippet":
                snippet[:800],

            "favicon":
                favicon
        })

        if len(results) >= max_results:
            break

    return results


# ============================================================
# IMAGE GENERATION
# ============================================================

async def generate_image(
    client: httpx.AsyncClient,
    prompt: str,
    steps: int = 4
):

    prompt = prompt.strip()

    if not prompt:

        raise RuntimeError(
            "Image prompt is empty."
        )

    # ========================================================
    # IMPORTANT:
    # FLUX.1 SCHNELL accepts prompt + steps.
    # NO SEED IS SENT.
    # ========================================================

    payload = {

        "prompt":
            prompt[:2048],

        "steps":
            max(
                1,
                min(
                    8,
                    int(steps)
                )
            )
    }

    response = await client.post(

        CF_IMAGE_URL,

        headers=
            cloudflare_headers(),

        json=
            payload,

        timeout=
            httpx.Timeout(
                connect=20,
                read=180,
                write=30,
                pool=20
            )
    )

    if response.status_code >= 400:

        raise RuntimeError(
            "Cloudflare Image Error: "
            + extract_cloudflare_error(
                response
            )
        )

    try:

        data = response.json()

    except Exception:

        raise RuntimeError(
            "Cloudflare Image API returned invalid JSON."
        )

    if not data.get(
        "success",
        False
    ):

        errors = data.get(
            "errors",
            []
        )

        raise RuntimeError(
            "Cloudflare Image Error: "
            + json.dumps(
                errors,
                ensure_ascii=False
            )
        )

    result = data.get(
        "result"
    )

    if not isinstance(
        result,
        dict
    ):

        raise RuntimeError(
            "Cloudflare returned an invalid image result."
        )

    image_base64 = result.get(
        "image"
    )

    if not isinstance(
        image_base64,
        str
    ) or not image_base64.strip():

        raise RuntimeError(
            "Cloudflare returned no image."
        )

    image_base64 = image_base64.strip()

    # جلوگیری از دوباره اضافه کردن data URI
    if image_base64.startswith(
        "data:image/"
    ):

        return image_base64

    return (
        "data:image/jpeg;base64,"
        + image_base64
    )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def homepage():

    index_file = (
        STATIC_DIR /
        "index.html"
    )

    if not index_file.exists():

        return {
            "service":
                APP_NAME,

            "version":
                APP_VERSION,

            "status":
                "ok"
        }

    return FileResponse(
        index_file
    )


# ============================================================
# HEALTH
# ============================================================

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

        "cloudflare_configured":
            cloudflare_configured(),

        "chat_model":
            CHAT_MODEL,

        "image_model":
            IMAGE_MODEL
    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/api/chat"
)
async def chat_endpoint(
    body: ChatRequest,
    request: Request
):

    if not cloudflare_configured():

        raise HTTPException(

            status_code=500,

            detail:
                "Cloudflare environment variables are not configured."
        )

    ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    if not check_rate_limit(ip):

        raise HTTPException(

            status_code=429,

            detail:
                "تعداد درخواست‌ها زیاد است. چند لحظه صبر کن."
        )

    messages = clean_messages(
        body.messages
    )

    if not messages:

        raise HTTPException(

            status_code=422,

            detail:
                "پیام خالی است."
        )

    if messages[-1].get(
        "role"
    ) != "user":

        raise HTTPException(

            status_code=422,

            detail:
                "آخرین پیام باید از طرف کاربر باشد."
        )

    async def event_stream():

        try:

            # =================================================
            # STATUS
            # =================================================

            yield make_sse(

                "status",

                {
                    "text":
                        "در حال بررسی درخواست..."
                }
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

                # =============================================
                # FIRST MODEL CALL
                # =============================================

                first_data = await cloudflare_chat(

                    client,

                    messages
                )

                choices = first_data.get(
                    "choices",
                    []
                )

                if not choices:

                    raise RuntimeError(
                        "Cloudflare returned no choices."
                    )

                choice = choices[0]

                assistant_message = (
                    choice.get(
                        "message"
                    )
                    or {}
                )

                tool_calls = (
                    assistant_message.get(
                        "tool_calls"
                    )
                    or []
                )

                # =============================================
                # NORMAL TEXT RESPONSE
                # =============================================

                if not tool_calls:

                    content = (
                        assistant_message.get(
                            "content"
                        )
                        or ""
                    )

                    content = str(
                        content
                    )

                    for part in re.findall(
                        r".{1,35}",
                        content,
                        flags=re.S
                    ):

                        yield make_sse(

                            "delta",

                            {
                                "text":
                                    part
                            }
                        )

                    yield make_sse(

                        "done",

                        {
                            "ok":
                                True
                        }
                    )

                    return

                # =============================================
                # TOOL ROUND
                # =============================================

                tool_messages = list(
                    messages
                )

                # نگه داشتن پیام assistant دارای tool_calls
                tool_messages.append(
                    assistant_message
                )

                executed_tools = 0

                for tool_call in tool_calls[:4]:

                    function = (
                        tool_call.get(
                            "function"
                        )
                        or {}
                    )

                    name = function.get(
                        "name"
                    )

                    arguments_raw = (
                        function.get(
                            "arguments"
                        )
                        or "{}"
                    )

                    try:

                        arguments = json.loads(
                            arguments_raw
                        )

                    except Exception:

                        arguments = {}

                    tool_call_id = (
                        tool_call.get(
                            "id"
                        )
                        or str(
                            uuid.uuid4()
                        )
                    )

                    tool_result = {}

                    # =========================================
                    # WEB SEARCH
                    # =========================================

                    if name == "web_search":

                        query = str(
                            arguments.get(
                                "query",
                                ""
                            )
                        ).strip()

                        if not query:

                            tool_result = {
                                "success":
                                    False,

                                "error":
                                    "Empty search query."
                            }

                        else:

                            yield make_sse(

                                "status",

                                {
                                    "text":
                                        (
                                            "در حال جستجو در وب: "
                                            + query
                                        )
                                }
                            )

                            results = await web_search(

                                client,

                                query,

                                int(
                                    arguments.get(
                                        "max_results",
                                        6
                                    )
                                )
                            )

                            yield make_sse(

                                "sources",

                                {
                                    "items":
                                        results
                                }
                            )

                            tool_result = {

                                "success":
                                    True,

                                "query":
                                    query,

                                "results":
                                    results
                            }

                        executed_tools += 1

                    # =========================================
                    # IMAGE GENERATION
                    # =========================================

                    elif name == "generate_image":

                        prompt = str(
                            arguments.get(
                                "prompt",
                                ""
                            )
                        ).strip()

                        steps = arguments.get(
                            "steps",
                            4
                        )

                        try:

                            steps = int(
                                steps
                            )

                        except Exception:

                            steps = 4

                        if not prompt:

                            tool_result = {

                                "success":
                                    False,

                                "error":
                                    "Empty image prompt."
                            }

                        else:

                            yield make_sse(

                                "status",

                                {
                                    "text":
                                        "در حال ساخت تصویر... 🎨"
                                }
                            )

                            image = await generate_image(

                                client,

                                prompt,

                                steps
                            )

                            # تصویر مستقیماً به فرانت‌اند
                            yield make_sse(

                                "image",

                                {
                                    "image":
                                        image,

                                    "prompt":
                                        prompt
                                }
                            )

                            tool_result = {

                                "success":
                                    True,

                                "image_generated":
                                    True,

                                "prompt":
                                    prompt
                            }

                        executed_tools += 1

                    # =========================================
                    # CHART
                    # =========================================

                    elif name == "create_chart":

                        labels = (
                            arguments.get(
                                "labels"
                            )
                            or []
                        )

                        values = (
                            arguments.get(
                                "values"
                            )
                            or []
                        )

                        if (
                            not labels
                            or
                            len(labels) != len(values)
                        ):

                            tool_result = {

                                "success":
                                    False,

                                "error":
                                    "Invalid chart data."
                            }

                        else:

                            chart = {

                                "title":
                                    str(
                                        arguments.get(
                                            "title",
                                            "نمودار"
                                        )
                                    ),

                                "type":
                                    str(
                                        arguments.get(
                                            "chart_type",
                                            "bar"
                                        )
                                    ),

                                "labels":
                                    labels,

                                "values":
                                    values,

                                "unit":
                                    str(
                                        arguments.get(
                                            "unit",
                                            ""
                                        )
                                    )
                            }

                            yield make_sse(

                                "chart",

                                chart
                            )

                            tool_result = {

                                "success":
                                    True,

                                "chart":
                                    chart
                            }

                        executed_tools += 1

                    else:

                        continue

                    # =========================================
                    # TOOL RESULT
                    # =========================================

                    tool_messages.append({

                        "role":
                            "tool",

                        "tool_call_id":
                            tool_call_id,

                        "content":
                            json.dumps(
                                tool_result,
                                ensure_ascii=False
                            )
                    })

                if executed_tools == 0:

                    raise RuntimeError(
                        "No supported tool was executed."
                    )

                # =============================================
                # FINAL MODEL CALL
                # =============================================

                yield make_sse(

                    "status",

                    {
                        "text":
                            "در حال آماده‌سازی پاسخ نهایی..."
                    }
                )

                final_data = await cloudflare_chat(

                    client,

                    tool_messages
                )

                final_choices = final_data.get(
                    "choices",
                    []
                )

                if not final_choices:

                    raise RuntimeError(
                        "No final response from Cloudflare."
                    )

                final_message = (
                    final_choices[0].get(
                        "message"
                    )
                    or {}
                )

                final_content = (
                    final_message.get(
                        "content"
                    )
                    or ""
                )

                final_content = str(
                    final_content
                ).strip()

                for part in re.findall(
                    r".{1,35}",
                    final_content,
                    flags=re.S
                ):

                    yield make_sse(

                        "delta",

                        {
                            "text":
                                part
                        }
                    )

                yield make_sse(

                    "done",

                    {
                        "ok":
                            True
                    }
                )

        except Exception as exc:

            # خطا را داخل SSE می‌فرستیم
            # تا frontend بتواند آن را نمایش دهد.

            logger_text = (
                f"CHAT ERROR: {type(exc).__name__}: {exc}"
            )

            print(
                logger_text,
                flush=True
            )

            yield make_sse(

                "error",

                {
                    "message":
                        str(exc)
                }
            )

    return StreamingResponse(

        event_stream(),

        media_type:
            "text/event-stream",

        headers={

            "Cache-Control":
                "no-cache",

            "Connection":
                "keep-alive",

            "X-Accel-Buffering":
                "no"
        }
    )


# ============================================================
# STATIC FILES
# ============================================================

if STATIC_DIR.exists():

    app.mount(

        "/static",

        StaticFiles(
            directory=STATIC_DIR
        ),

        name="static"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        ),

        reload=False
)
