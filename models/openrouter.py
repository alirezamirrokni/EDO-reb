import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .groq import GroqConfig, GroqLLM


class OpenRouterLLM(GroqLLM):
    """OpenRouter-hosted LLM with the same prompting, throttling and retry behaviour as GroqLLM."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, cfg: Dict[str, Any]):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (put it in .env).")
        cfg = dict(cfg)
        self.provider_routing = cfg.pop("provider_routing", None)
        self.usage_log = cfg.pop("usage_log", "outputs/openrouter_usage.jsonl")
        self.cfg = GroqConfig(**cfg)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        self._last_call_time = 0.0
        from .groq import DEFAULT_SYSTEM_PROMPT

        self.system_prompt = self.cfg.system_prompt if self.cfg.system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        print(
            f"[openrouter] model={self.cfg.model_name} provider_routing={self.provider_routing} "
            f"max_output_tokens={self.cfg.max_output_tokens}"
        )

    def _log_usage(self, usage: Dict[str, Any]) -> None:
        path = Path(self.usage_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"time": time.time(), "model": self.cfg.model_name, **usage}) + "\n")

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                self._throttle()
                no_think = str(self.cfg.reasoning_effort).lower() == "none" and "qwen3" in self.cfg.model_name.lower()
                body: Dict[str, Any] = {
                    "model": self.cfg.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt + (" /no_think" if no_think else "")},
                    ],
                    "temperature": self.cfg.temperature,
                    "max_tokens": self.cfg.max_output_tokens,
                    "usage": {"include": True},
                }
                if self.cfg.reasoning_effort is not None:
                    if str(self.cfg.reasoning_effort).lower() == "none":
                        body["reasoning"] = {"enabled": False, "exclude": True}
                    else:
                        body["reasoning"] = {"effort": self.cfg.reasoning_effort, "exclude": True}
                if self.provider_routing:
                    body["provider"] = self.provider_routing
                resp = self.session.post(self.URL, data=json.dumps(body), timeout=self.cfg.request_timeout)
                self._last_call_time = time.time()
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(str(data["error"])[:300])
                if data.get("usage"):
                    self._log_usage(data["usage"])
                text = data["choices"][0]["message"].get("content")
                return str(text if text is not None else "").strip()
            except Exception as exc:
                last_err = exc
                wait = self.cfg.retry_sleep * attempt
                if "429" in str(exc) or "rate" in str(exc).lower():
                    wait = max(wait, 10.0)
                time.sleep(wait)
        raise RuntimeError(f"OpenRouter call failed after {self.cfg.retry_attempts} attempts: {last_err}")
