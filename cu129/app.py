"""HTTP API for analyzing conversation JSON with an existing vLLM server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError


logger = logging.getLogger("uvicorn.error")
NGROK_API_URL = os.getenv(
    "NGROK_API_URL",
    "http://127.0.0.1:4040/api/tunnels",
)


def _get_ngrok_public_url() -> str | None:
    """Return the HTTPS tunnel URL reported by the local ngrok agent."""
    try:
        with urlopen(NGROK_API_URL, timeout=1) as response:
            payload = json.load(response)
    except (OSError, URLError, ValueError):
        return None

    urls = [
        tunnel.get("public_url")
        for tunnel in payload.get("tunnels", [])
        if isinstance(tunnel, dict)
    ]
    return next(
        (url for url in urls if isinstance(url, str) and url.startswith("https://")),
        next((url for url in urls if isinstance(url, str)), None),
    )


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Optionally run an ngrok tunnel for the lifetime of the API process."""
    if not os.getenv("NGROK_AUTHTOKEN", "").strip():
        application.state.ngrok_public_url = None
        logger.info("ngrok is disabled; the API is available only on its local port.")
        yield
        return

    app_port = int(os.getenv("APP_PORT", "7000"))
    try:
        ngrok_process = subprocess.Popen(
            ["ngrok", "http", f"http://127.0.0.1:{app_port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ngrok is not installed or is not on PATH.") from exc

    try:
        public_url = None
        for _ in range(30):
            if ngrok_process.poll() is not None:
                raise RuntimeError(
                    "ngrok exited before creating a tunnel. Configure an auth token "
                    "with NGROK_AUTHTOKEN or 'ngrok config add-authtoken'."
                )
            public_url = await asyncio.to_thread(_get_ngrok_public_url)
            if public_url:
                break
            await asyncio.sleep(0.5)

        if public_url is None:
            raise RuntimeError("ngrok did not report a public URL within 15 seconds.")

        application.state.ngrok_public_url = public_url
        logger.info("ngrok public URL: %s", public_url)
        yield
    finally:
        if ngrok_process.poll() is None:
            ngrok_process.terminate()
            try:
                await asyncio.to_thread(ngrok_process.wait, 10)
            except subprocess.TimeoutExpired:
                ngrok_process.kill()
                await asyncio.to_thread(ngrok_process.wait)


class AnalysisRequest(BaseModel):
    """The instructions and conversation payload sent to the model."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    analyzing_prompt: str = Field(
        min_length=1,
        description="Analysis instructions inserted as the system message.",
    )
    json_input: dict[str, Any] | list[Any] = Field(
        alias="input",
        description="Conversation dialogue and its metadata.",
    )


class AnalysisResponse(BaseModel):
    result: Any


class NgrokStatus(BaseModel):
    public_url: str


app = FastAPI(
    title="vLLM Conversation Analyzer",
    version="1.0.0",
    lifespan=lifespan,
)

vllm_client = AsyncOpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.getenv("VLLM_API_KEY", "not-needed"),
    timeout=float(os.getenv("VLLM_TIMEOUT", "300")),
)


@app.get("/ngrok", response_model=NgrokStatus)
async def ngrok_status() -> NgrokStatus:
    """Return the public URL for this running API instance."""
    public_url = getattr(app.state, "ngrok_public_url", None)
    if public_url is None:
        raise HTTPException(status_code=503, detail="ngrok tunnel is not ready.")
    return NgrokStatus(public_url=public_url)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze one JSON conversation and return a JSON-native model result."""
    messages = [
        {"role": "system", "content": request.analyzing_prompt},
        {
            "role": "user",
            "content": json.dumps(request.json_input, ensure_ascii=False),
        },
    ]
    try:
        completion = await vllm_client.chat.completions.create(
            model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B"),
            messages=messages,
            temperature=float(os.getenv("VLLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "2048")),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "analysis_response",
                    "schema": {
                        "type": "object",
                        "properties": {"result": {}},
                        "required": ["result"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        generated_text = completion.choices[0].message.content
        if generated_text is None:
            raise HTTPException(status_code=502, detail="vLLM returned no content.")
        result = AnalysisResponse.model_validate_json(generated_text)
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to the vLLM server.",
        ) from exc
    except APIStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"vLLM returned HTTP {exc.status_code}.",
        ) from exc
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(
            status_code=502,
            detail="The model returned an invalid result structure.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return result
