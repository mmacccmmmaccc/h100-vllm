"""HTTP API for analyzing conversation JSON with an existing vLLM server."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError


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


app = FastAPI(title="vLLM Conversation Analyzer", version="1.0.0")

vllm_client = AsyncOpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.getenv("VLLM_API_KEY", "not-needed"),
    timeout=float(os.getenv("VLLM_TIMEOUT", "300")),
)


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
