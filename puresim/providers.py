"""Model providers for the agent under test.

The sandbox is deliberately model-agnostic: the product is the safety report
card, not any one vendor's model. A provider's only job is to turn a system
prompt plus an observation into raw response text. Parsing, validation,
clamping, and scoring all happen identically downstream, so results across
providers are directly comparable.

Four providers ship here:

* ``AnthropicProvider`` — Claude, via the official SDK.
* ``GeminiProvider``   — Google AI Studio REST API (has a real free tier).
* ``OllamaProvider``   — a local Ollama server; free, offline, open weights.
* ``OpenAICompatibleProvider`` — anything speaking the OpenAI chat-completions
  shape. Covers OpenAI itself, and also Groq and OpenRouter by pointing
  ``base_url`` elsewhere.

Everything except Anthropic is spoken over stdlib ``urllib`` so the project
picks up no new dependencies.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

# JSON Schema for the action, shared by every provider that supports
# constrained decoding.
ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "amount": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "amount", "reasoning"],
    "additionalProperties": False,
}

DEFAULT_TIMEOUT = 60


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a response.

    Always caught by the agent, which falls back to HOLD — a provider failure
    must never end a run.
    """


class Provider(ABC):
    """Turns a system prompt plus an observation into raw response text."""

    #: Which lab or runtime this model comes from, for the comparison table.
    vendor: str = "unknown"

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def label(self) -> str:
        return f"{self.vendor}:{self.model}"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response, or raise ProviderError."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: int,
    max_retries: int = 4,
) -> dict:
    """POST JSON and return the decoded response. Raises ProviderError on failure.

    Retries rate limits (429) and transient server errors with exponential
    backoff, honouring ``Retry-After`` when the server sends one. Free tiers cap
    requests per minute, and a run fires one request per tick in a tight loop,
    so without this a long run reliably dies partway through.
    """
    data = json.dumps(payload).encode("utf-8")
    delay = 2.0
    last_error = "unknown"

    for attempt in range(max_retries):
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        # Some providers sit behind Cloudflare, which blocks urllib's default
        # "Python-urllib/x.y" User-Agent as a bot signature (HTTP 403, Cloudflare
        # error 1010). A plain browser-like UA clears it.
        request.add_header("User-Agent", "Mozilla/5.0 (compatible; PureSim/1.0)")
        for key, value in headers.items():
            request.add_header(key, value)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code not in (429, 500, 502, 503, 504) or attempt == max_retries - 1:
                raise ProviderError(last_error) from exc
            # The server usually tells us exactly how long to wait; trust it.
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            print(f"    [rate limited, retrying in {wait:.1f}s]")
            time.sleep(min(wait, 30.0))
            delay *= 2
        except urllib.error.URLError as exc:
            raise ProviderError(f"connection failed: {exc.reason}") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"bad response: {exc}") from exc

    raise ProviderError(last_error)


class AnthropicProvider(Provider):
    """Claude, via the official Anthropic SDK.

    Uses structured outputs so the model is constrained to the action schema,
    and a low effort level to keep per-tick latency and cost down — a
    BUY/SELL/HOLD call does not need deep reasoning.
    """

    vendor = "anthropic"

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        effort: str = "low",
        max_tokens: int = 2000,
    ) -> None:
        super().__init__(model)
        self.effort = effort
        self.max_tokens = max_tokens

        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("anthropic SDK not installed (pip install anthropic)") from exc

        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:  # noqa: BLE001 - surfaced as a provider failure
            raise ProviderError(f"client init failed: {exc}") from exc

    def complete(self, system: str, user: str) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": ACTION_SCHEMA},
                },
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - any SDK error is a tick failure
            raise ProviderError(str(exc)) from exc

        if response.stop_reason == "refusal":
            raise ProviderError("model refused the request")

        return next((b.text for b in response.content if b.type == "text"), "")


class GeminiProvider(Provider):
    """Google Gemini via the AI Studio REST API.

    Reads ``GEMINI_API_KEY`` (falling back to ``GOOGLE_API_KEY``). Uses
    ``responseSchema`` for constrained JSON decoding.
    """

    vendor = "google"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(model)
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not self.api_key:
            raise ProviderError("set GEMINI_API_KEY (free tier: aistudio.google.com)")

    def complete(self, system: str, user: str) -> str:
        # Gemini's schema dialect rejects additionalProperties, so send a
        # trimmed copy rather than the shared one.
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                "amount": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["action", "amount", "reasoning"],
        }
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        url = self.ENDPOINT.format(model=self.model)
        body = _post_json(url, payload, {"x-goog-api-key": self.api_key}, self.timeout)

        try:
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(body)[:200]}") from exc


class OllamaProvider(Provider):
    """A locally running Ollama server. Free, offline, open weights.

    Requires ``ollama serve`` to be running and the model pulled, e.g.
    ``ollama pull llama3.2``.
    """

    vendor = "ollama"

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(model)
        self.host = host.rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            # Ollama accepts a JSON Schema here to constrain decoding.
            "format": ACTION_SCHEMA,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        body = _post_json(f"{self.host}/api/chat", payload, {}, self.timeout)

        try:
            return body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(body)[:200]}") from exc


class OpenAICompatibleProvider(Provider):
    """Any endpoint speaking the OpenAI chat-completions shape.

    Covers OpenAI itself, and also Groq and OpenRouter by changing ``base_url``
    and the environment variable holding the key. Groq's free tier makes it a
    good zero-cost way to add another vendor to the comparison.
    """

    vendor = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        vendor: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if vendor:
            self.vendor = vendor

        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ProviderError(f"set {api_key_env}")

    def complete(self, system: str, user: str) -> str:
        # OpenAI-compatible APIs (including Groq) require the literal word
        # "json" somewhere in the prompt when response_format is json_object,
        # or the request 400s — independent of whether the schema itself
        # mentions JSON. Enforced here rather than baked into the shared system
        # prompt, since only this provider needs it.
        system = system + "\n\nRespond with a single JSON object only."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # json_object is the widely supported mode; strict schemas vary by
            # endpoint, and the downstream parser validates regardless.
            "response_format": {"type": "json_object"},
        }
        body = _post_json(
            f"{self.base_url}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
        )

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(body)[:200]}") from exc


def groq_provider(model: str = "llama-3.3-70b-versatile", **kwargs) -> OpenAICompatibleProvider:
    """Groq — OpenAI-compatible, free tier, serves open-weights models fast.

    Groq is a hardware/inference company, not a model lab — it just runs other
    labs' open-weights models very quickly. Not to be confused with xAI's Grok
    model, which is a different product with no free tier.
    """
    return OpenAICompatibleProvider(
        model=model,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        vendor="groq",
        **kwargs,
    )


#: Short names usable from the CLI. Each entry builds a provider on demand so
#: that a missing key for one vendor never blocks running another.
#:
#: Every entry below is free. Groq's free tier serves open-weights models from
#: several different labs, so the comparison table can show Meta, Google,
#: Alibaba, and OpenAI's own open release without a single paid key:
#:
#:   groq-llama    Meta      (Llama 3.3 70B)
#:   groq-gemma    Google    (Gemma 2 9B)
#:   groq-qwen     Alibaba   (Qwen 2.5 32B)
#:   groq-gptoss   OpenAI    (gpt-oss 20B, open-weights)
#:   ollama        anyone    (fully local — no key, no network, no rate limit)
PROVIDER_FACTORIES = {
    "groq-llama": lambda: groq_provider(model="llama-3.3-70b-versatile"),
    "groq-gemma": lambda: groq_provider(model="gemma2-9b-it"),
    "groq-qwen": lambda: groq_provider(model="qwen2.5-32b-instruct"),
    "groq-gptoss": lambda: groq_provider(model="openai/gpt-oss-20b"),
    "ollama": lambda: OllamaProvider(model="llama3.2"),
    # Paid — kept available but not part of the default free lineup.
    "claude-opus": lambda: AnthropicProvider(model="claude-opus-5"),
    "claude-sonnet": lambda: AnthropicProvider(model="claude-sonnet-5"),
    "claude-haiku": lambda: AnthropicProvider(model="claude-haiku-4-5"),
    "gemini": lambda: GeminiProvider(model="gemini-2.0-flash"),
    "openai": lambda: OpenAICompatibleProvider(model="gpt-4o-mini"),
}

#: The lineup `run_safety.py --compare` uses when no models are named — every
#: entry free, spanning four different labs.
FREE_COMPARISON = ["groq-llama", "groq-gemma", "groq-qwen", "groq-gptoss"]


def build_provider(name: str) -> Provider:
    """Build a provider by CLI short name. Raises ProviderError if unavailable."""
    factory = PROVIDER_FACTORIES.get(name)
    if factory is None:
        known = ", ".join(sorted(PROVIDER_FACTORIES))
        raise ProviderError(f"unknown provider {name!r}; choose from: {known}")
    return factory()
