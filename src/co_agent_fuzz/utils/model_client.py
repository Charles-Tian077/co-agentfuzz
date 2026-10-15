from __future__ import annotations

import asyncio
import ipaddress
import threading
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse

import httpx
from openai import OpenAI


class ChatBackend(Protocol):
    async def complete(self, messages: List[Dict[str, str]], max_tokens: int = 512,
                       temperature: float = 0.0) -> str:
        ...


def _is_loopback(url: Optional[str]) -> bool:
    if not url:
        return False
    hostname = urlparse(url).hostname
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class OpenAIChatBackend:
    def __init__(self, model: str, api_key: str = "ollama", base_url: Optional[str] = None,
                 timeout_seconds: float = 120.0, max_concurrency: int = 4):
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout_seconds}
        if base_url:
            kwargs["base_url"] = base_url
        if _is_loopback(base_url):
            kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout_seconds)
        self.client = OpenAI(**kwargs)
        self.model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.request_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._usage_lock = threading.Lock()

    async def complete(self, messages: List[Dict[str, str]], max_tokens: int = 512,
                       temperature: float = 0.0) -> str:
        async with self._semaphore:
            return await asyncio.to_thread(
                self._complete_sync, messages, max_tokens, temperature)

    def _complete_sync(self, messages: List[Dict[str, str]], max_tokens: int,
                       temperature: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
        )
        usage = getattr(response, "usage", None)
        with self._usage_lock:
            self.request_count += 1
            if usage is not None:
                self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return response.choices[0].message.content or ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def usage(self) -> Dict[str, int]:
        return {
            "requests": self.request_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def close(self) -> None:
        self.client.close()
