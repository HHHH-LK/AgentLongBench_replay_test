"""单次流式请求的 TTFT/TPOT/E2E 计时。

请求格式严格遵循 OpenAI Chat Completions, 兼容:
  - vLLM serve  (`vllm.entrypoints.openai.api_server`)
  - sglang serve
  - Anthropic API (走兼容层时)
  - 其他 OpenAI 兼容服务

KV 复用前提:
  调用方必须保证同一会话的多个请求**串行**打到**同一个 base_url 服务实例**,
  且后续请求的 messages 是前一个的严格前缀 + 新增尾巴, 这样底层
  prefix cache (vLLM 默认 / LMCache) 才能命中。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class TimedResult:
    raw_response: str
    ttft_ms: Optional[float]
    e2e_ms: Optional[float]
    decode_ms: Optional[float]
    tpot_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    is_streaming: bool
    error: Optional[str] = None


def timed_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_key: str = "EMPTY",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 1200,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> TimedResult:
    """流式发一次 chat completion 请求, 采集 TTFT 等指标。"""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # 让最后一个 chunk 带 usage, 这样能拿到 prompt/completion tokens
        "stream_options": {"include_usage": True},
    }
    if extra_payload:
        payload.update(extra_payload)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    t0 = time.perf_counter()
    first_token_at: Optional[float] = None
    content_buf: List[str] = []
    usage: Dict[str, Any] = {}
    error: Optional[str] = None

    try:
        with requests.post(
            url, json=payload, headers=headers, stream=True, timeout=timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if not line_str.startswith("data: "):
                    continue
                data = line_str[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        content_buf.append(piece)

                u = chunk.get("usage")
                if u:
                    usage = u
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    ttft_ms = (first_token_at - t0) * 1000.0 if first_token_at is not None else None
    decode_ms = max(e2e_ms - ttft_ms, 0.0) if ttft_ms is not None else None
    n_completion = usage.get("completion_tokens")
    tpot_ms = (
        decode_ms / max(n_completion - 1, 1)
        if (decode_ms is not None and n_completion and n_completion > 1)
        else None
    )

    return TimedResult(
        raw_response="".join(content_buf),
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        decode_ms=decode_ms,
        tpot_ms=tpot_ms,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=n_completion,
        is_streaming=True,
        error=error,
    )
