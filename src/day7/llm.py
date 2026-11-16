"""Minimal OpenAI-compatible chat client (default: Google Gemini's OpenAI endpoint).

Env (put in ROOT/.env, never commit it):
  GEMINI_API_KEY or LLM_API_KEY
  LLM_BASE_URL   default https://generativelanguage.googleapis.com/v1beta/openai
  LLM_MODEL      default gemini-3.5-flash
  LLM_REASONING_EFFORT  optional (e.g. none/low) passed as reasoning_effort
  LLM_PRICE_IN_PER_M / LLM_PRICE_OUT_PER_M  USD per 1M tokens for the cost estimate
  LLM_MIN_INTERVAL_S    min seconds between calls (free-tier rate limit), default 4
Errors are raised after retries; no fake answers.
"""
import os
import time
from typing import Any

import httpx

from .common import load_env


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self) -> None:
        load_env()
        env = os.environ.get
        self.provider = (env("LLM_PROVIDER") or "gemini").strip().lower()
        if self.provider == "openai":
            # OPENAI_* settings are used only when LLM_PROVIDER=openai
            self.api_key = env("OPENAI_API_KEY")
            self.base = (env("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
            self.model = env("OPENAI_MODEL") or "gpt-4.1-mini"
            self.effort = env("OPENAI_REASONING_EFFORT") or None
            self.price_in = float(env("OPENAI_PRICE_IN_PER_M") or "0.40")
            self.price_out = float(env("OPENAI_PRICE_OUT_PER_M") or "1.60")
            self.min_interval = float(env("OPENAI_MIN_INTERVAL_S") or "0")
        elif self.provider == "gemini":
            self.api_key = env("LLM_API_KEY") or env("GEMINI_API_KEY")
            self.base = (env("LLM_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
            self.model = env("LLM_MODEL") or "gemini-3.5-flash"
            self.effort = env("LLM_REASONING_EFFORT") or None
            self.price_in = float(env("LLM_PRICE_IN_PER_M") or "0.30")
            self.price_out = float(env("LLM_PRICE_OUT_PER_M") or "2.50")
            self.min_interval = float(env("LLM_MIN_INTERVAL_S") or "4")
        else:
            raise LLMError(f"LLM_PROVIDER must be gemini or openai, got {self.provider!r}")
        if not self.api_key:
            raise LLMError(f"No API key for provider {self.provider}: set "
                           f"{'OPENAI_API_KEY' if self.provider == 'openai' else 'GEMINI_API_KEY'} in .env")
        self.send_temperature = True
        self._last = 0.0
        self.http = httpx.Client(timeout=120)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | None = None, retries: int = 6) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.send_temperature:
            body["temperature"] = 0
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        if self.effort:
            body["reasoning_effort"] = self.effort
        delay = 10.0
        for attempt in range(retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            self._last = time.monotonic()
            try:
                r = self.http.post(f"{self.base}/chat/completions", json=body,
                                   headers={"Authorization": f"Bearer {self.api_key}"})
            except httpx.HTTPError as exc:
                err = f"network: {exc!r}"
            else:
                if r.status_code == 200:
                    data = r.json()
                    usage = data.get("usage") or {}
                    pin, pout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                    return {"message": data["choices"][0]["message"],
                            "finish_reason": data["choices"][0].get("finish_reason"),
                            "usage": {"prompt_tokens": pin, "completion_tokens": pout},
                            "cost_usd_est": (pin * self.price_in + pout * self.price_out) / 1e6,
                            "latency_s": round(time.perf_counter() - t0, 2), "attempts": attempt + 1}
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code == 400 and "temperature" in r.text and self.send_temperature:
                    # reasoning models (e.g. gpt-5 family) only accept the default temperature
                    self.send_temperature = False
                    body.pop("temperature", None)
                    print("  [llm] model rejects temperature=0 -> using provider default", flush=True)
                    continue
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise LLMError(err)
            if attempt == retries:
                raise LLMError(f"giving up after {retries + 1} attempts: {err}")
            print(f"  [llm retry {attempt + 1}] {err[:120]} -> sleep {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 90)
        raise LLMError("unreachable")
