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
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

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

# Same shape, but requires at least one character of reasoning. Only used
# where the endpoint actually enforces the schema via constrained decoding
# (see `use_strict_schema` on OpenAICompatibleProvider) — under json_object
# mode a minLength constraint would just be ignored, so it isn't worth
# adding there.
STRICT_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "amount": {"type": "number"},
        "reasoning": {"type": "string", "minLength": 1},
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


# Minimum gap enforced between requests to the same host, across every
# provider instance that shares it. A single --compete run can have several
# competing agents all calling the same vendor (e.g. three groq-* models),
# and without a shared floor here each agent's retry-with-backoff only
# smooths out its own request stream — the *combined* rate from several
# agents firing back-to-back within the same tick still blows through a free
# tier's requests-per-minute cap, and every retry burns another slot of that
# same cap, so the whole run degrades into a 429 storm instead of recovering.
_throttle_lock = threading.Lock()
_last_request_at: dict[str, float] = {}


def _throttle(host: str, min_interval: float) -> None:
    if min_interval <= 0:
        return
    with _throttle_lock:
        now = time.monotonic()
        wait = min_interval - (now - _last_request_at.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_request_at[host] = time.monotonic()


def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: int,
    max_retries: int = 4,
    min_interval: float = 0.0,
) -> dict:
    """POST JSON and return the decoded response. Raises ProviderError on failure.

    Retries rate limits (429) and transient server errors with exponential
    backoff, honouring ``Retry-After`` when the server sends one. ``min_interval``
    additionally spaces out requests to the same host *before* they're sent,
    shared across every provider instance hitting that host (see ``_throttle``)
    — retries alone aren't enough once more than one agent shares a vendor.
    """
    host = urlparse(url).netloc
    data = json.dumps(payload).encode("utf-8")
    delay = 2.0
    last_error = "unknown"

    for attempt in range(max_retries):
        _throttle(host, min_interval)
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
        # AI Studio's free tier is roughly 15 requests/minute; stay under it.
        body = _post_json(
            url, payload, {"x-goog-api-key": self.api_key}, self.timeout, min_interval=4.5
        )

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
        min_request_interval: float = 0.0,
        use_strict_schema: bool = False,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Shared per-host, so every model behind the same base_url (e.g. every
        # groq-* competitor in a --compete run) is spaced out together rather
        # than each independently believing it has the full rate limit.
        self.min_request_interval = min_request_interval
        # json_object mode only guarantees syntactically valid JSON — a model
        # can satisfy it with `"reasoning": ""` and nothing catches that. Only
        # a few endpoints/models support real constrained decoding against a
        # schema (currently: OpenAI's gpt-oss family on Groq); this opts in.
        self.use_strict_schema = use_strict_schema
        if vendor:
            self.vendor = vendor

        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise ProviderError(f"set {api_key_env}")

    def complete(self, system: str, user: str) -> str:
        if self.use_strict_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "trade_action",
                    "strict": True,
                    "schema": STRICT_ACTION_SCHEMA,
                },
            }
        else:
            # OpenAI-compatible APIs (including Groq) require the literal
            # word "json" somewhere in the prompt when response_format is
            # json_object, or the request 400s — independent of whether the
            # schema itself mentions JSON. Enforced here rather than baked
            # into the shared system prompt, since only this mode needs it.
            system = system + "\n\nRespond with a single JSON object only."
            response_format = {"type": "json_object"}

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": response_format,
        }
        body = _post_json(
            f"{self.base_url}/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
            self.timeout,
            min_interval=self.min_request_interval,
        )

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(body)[:200]}") from exc


def groq_provider(
    model: str = "llama-3.3-70b-versatile", min_request_interval: float = 2.5, **kwargs
) -> OpenAICompatibleProvider:
    """Groq — OpenAI-compatible, free tier, serves open-weights models fast.

    Groq is a hardware/inference company, not a model lab — it just runs other
    labs' open-weights models very quickly. Not to be confused with xAI's Grok
    model, which is a different product with no free tier.

    ``min_request_interval`` defaults to a conservative 2.5s: the free tier's
    requests-per-minute cap is shared across every groq-* model on the same
    key, so in a --compete run with several groq-* competitors this spacing
    applies across all of them together (they share one host-keyed throttle),
    not 2.5s per model.
    """
    return OpenAICompatibleProvider(
        model=model,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        vendor="groq",
        min_request_interval=min_request_interval,
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
#:   groq-qwen     Alibaba   (Qwen 2.5 32B)
#:   groq-gptoss   OpenAI    (gpt-oss 20B, open-weights)
#:   ollama        anyone    (fully local — no key, no network, no rate limit)
#:
#: No Gemma entry: Google pulled Gemma from Groq's lineup entirely (confirmed
#: against console.groq.com/docs/models, 2026-08-13) — there is no working
#: model ID to fall back to, unlike the Qwen ID fix above. If Groq adds a
#: Gemma model back, add it here with a "groq-gemma" key.
PROVIDER_FACTORIES = {
    "groq-llama": lambda: groq_provider(model="llama-3.3-70b-versatile"),
    "groq-qwen": lambda: groq_provider(model="qwen-2.5-32b"),
    # Only gpt-oss on Groq supports strict schema-constrained decoding (as of
    # 2026-08-13, per console.groq.com/docs/structured-outputs) — Llama and
    # Qwen there still get the best-effort json_object mode above.
    "groq-gptoss": lambda: groq_provider(model="openai/gpt-oss-20b", use_strict_schema=True),
    "ollama": lambda: OllamaProvider(model="llama3.2"),
    # Paid — kept available but not part of the default free lineup.
    "claude-opus": lambda: AnthropicProvider(model="claude-opus-5"),
    "claude-sonnet": lambda: AnthropicProvider(model="claude-sonnet-5"),
    "claude-haiku": lambda: AnthropicProvider(model="claude-haiku-4-5"),
    "gemini": lambda: GeminiProvider(model="gemini-2.0-flash"),
    "openai": lambda: OpenAICompatibleProvider(model="gpt-4o-mini"),
}

#: The lineup `run_safety.py --compare` uses when no models are named — every
#: entry free, spanning three different labs.
FREE_COMPARISON = ["groq-llama", "groq-qwen", "groq-gptoss"]


def build_provider(name: str) -> Provider:
    """Build a provider by CLI short name. Raises ProviderError if unavailable."""
    factory = PROVIDER_FACTORIES.get(name)
    if factory is None:
        known = ", ".join(sorted(PROVIDER_FACTORIES))
        raise ProviderError(f"unknown provider {name!r}; choose from: {known}")
    return factory()
