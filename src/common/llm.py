"""Separate generation/embedding transports with durable, replayable API responses."""

import hashlib
import math
import os
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import anthropic
import httpx
import tiktoken
from dotenv import dotenv_values

from common.artifacts import checkpoint, digest, now, read, save, verified

SATURATION_ATTEMPTS = 12  # about six minutes of capped backoff
REASONING_TIMEOUT = 600  # seconds; a high-effort answer over the full history takes minutes
REASONING_OUTPUT_FLOOR = 16000  # output tokens, reasoning included, when reasoning is on


class UnservedError(ValueError):
    """An HTTP-200 body that carries only an error: a router relayed an unserved request."""

    def __init__(self, error):
        super().__init__(error.get("message", "error body"))
        self.status = error.get("code") if isinstance(error.get("code"), int) else None


class StatusError(ValueError):
    """An HTTP error status raised by a provider SDK."""

    def __init__(self, status, message=""):
        super().__init__(f"HTTP {status} {message}"[:500])
        self.status = status


TRANSLATED_FIELDS = {"model", "messages", "max_tokens", "max_completion_tokens", "reasoning_effort",
                     "response_format", "service_tier", "seed", "temperature"}


def split_messages(payload):
    unknown = set(payload) - TRANSLATED_FIELDS
    if unknown:
        raise ValueError(f"No translation for request fields {sorted(unknown)}")
    if any(not isinstance(m.get("content"), str) for m in payload["messages"]):
        raise ValueError("Only text message content is translated")
    return [{"role": m["role"], "content": m["content"]} for m in payload["messages"]]


def responses_request(payload):
    """A chat completions payload as an OpenAI Responses request. Responses has no seed."""
    wire = {"model": payload["model"], "input": split_messages(payload), "store": False}
    limit = payload.get("max_completion_tokens", payload.get("max_tokens"))
    if limit is not None:
        wire["max_output_tokens"] = limit
    if payload.get("reasoning_effort") is not None:
        wire["reasoning"] = {"effort": payload["reasoning_effort"]}
    response_format = payload.get("response_format")
    if response_format and response_format["type"] == "json_schema":
        wire["text"] = {"format": {"type": "json_schema", **response_format["json_schema"]}}
    elif response_format and response_format["type"] != "text":
        wire["text"] = {"format": {"type": response_format["type"]}}
    if "service_tier" in payload:
        wire["service_tier"] = payload["service_tier"]
    return wire


def responses_to_chat(raw):
    text = "".join(c.get("text", "") for item in raw.get("output") or [] if item.get("type") == "message"
                   for c in item.get("content") or [] if c.get("type") == "output_text")
    reason = (raw.get("incomplete_details") or {}).get("reason")
    finish = "length" if reason == "max_output_tokens" else "stop" if raw.get("status") == "completed" else raw.get("status")
    usage = raw.get("usage") or {}
    return chat_completion(raw.get("id"), raw.get("created_at"), raw.get("model"), text, finish,
                           usage.get("input_tokens"), usage.get("output_tokens"),
                           (usage.get("input_tokens_details") or {}).get("cached_tokens"),
                           (usage.get("output_tokens_details") or {}).get("reasoning_tokens"))


def chat_completion(id, created, model, text, finish, input_tokens, output_tokens, cached, reasoning):
    """A complete chat.completion object, which SDK-based callers (Graphiti) validate."""
    return {"id": id, "object": "chat.completion", "created": int(created or 0), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                      "total_tokens": (input_tokens or 0) + (output_tokens or 0),
                      "prompt_tokens_details": {"cached_tokens": cached},
                      "completion_tokens_details": {"reasoning_tokens": reasoning}}}


def anthropic_request(payload):
    """A chat completions payload as an Anthropic Messages request. Claude Opus 5.5 always thinks;
    effort sets how much. It takes no temperature or seed, and json_object has no Messages
    equivalent, so that format is left to the prompt and the caller's parser."""
    if payload.get("reasoning_effort") in (None, "none"):
        raise ValueError("Claude Opus 5.5 cannot turn thinking off; set a reasoning effort")
    messages = split_messages(payload)
    wire = {"model": payload["model"],
            "max_tokens": payload.get("max_completion_tokens") or payload.get("max_tokens") or REASONING_OUTPUT_FLOOR,
            "messages": [m for m in messages if m["role"] != "system"],
            "output_config": {"effort": payload["reasoning_effort"]}}
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    if system:
        wire["system"] = system
    response_format = payload.get("response_format")
    if response_format and response_format["type"] == "json_schema":
        wire["output_config"]["format"] = {"type": "json_schema", "schema": closed(response_format["json_schema"]["schema"])}
    return wire


def closed(schema):
    """Anthropic structured outputs require additionalProperties: false on every object. That only
    forbids undeclared keys; which keys are required is unchanged."""
    if isinstance(schema, list):
        return [closed(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    schema = {k: closed(v) for k, v in schema.items()}
    if schema.get("type") == "object" or "properties" in schema:
        schema.setdefault("additionalProperties", False)
    return schema


def anthropic_to_chat(raw):
    text = "".join(b.get("text", "") for b in raw.get("content") or [] if b.get("type") == "text")
    finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}.get(raw.get("stop_reason"), raw.get("stop_reason"))
    usage = raw.get("usage") or {}
    read_tokens, write_tokens = usage.get("cache_read_input_tokens") or 0, usage.get("cache_creation_input_tokens") or 0
    # Input counts every prompt token, cached or not, as with the OpenAI providers.
    return chat_completion(raw.get("id"), 0, raw.get("model"), text, finish,
                           usage.get("input_tokens", 0) + read_tokens + write_tokens, usage.get("output_tokens"),
                           read_tokens, (usage.get("output_tokens_details") or {}).get("thinking_tokens"))

DEFAULT_ENV = Path(__file__).resolve().parents[2] / ".env"


@lru_cache(maxsize=1)
def context_encoding():
    return tiktoken.get_encoding("cl100k_base")


def context_metrics(text):
    return {
        "context_tokens": len(context_encoding().encode(text)),
        "context_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def usage_record(response, kind, model, role=None):
    usage = response.get("usage") or {}
    cost = usage.get("cost_usd", response.get("cost_usd"))
    currency = usage.get("currency", response.get("currency"))
    reported = usage.get("cost", response.get("cost"))
    if cost is None and currency == "USD" and isinstance(reported, (int, float)):
        cost = reported
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(cost)
    ):
        cost = None
    return {
        "kind": kind,
        "role": role,
        "model": response.get("model", model),
        "requested_model": model,
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "cached_input_tokens": (
            usage.get("prompt_tokens_details")
            or usage.get("input_tokens_details")
            or {}
        ).get("cached_tokens"),
        "reasoning_tokens": (
            usage.get("completion_tokens_details")
            or usage.get("output_tokens_details")
            or {}
        ).get("reasoning_tokens"),
        "cost_usd": cost,
        "reported_cost": reported,
        "reported_currency": currency,
        "raw_usage": usage,
        "response_id": response.get("id"),
    }


def api_root(url):
    url = url.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/embeddings"):
        url = url.removesuffix(suffix)
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.query
    ):
        raise ValueError("Expected a base URL without credentials or query parameters")
    return url + "/v1" if not parsed.path else url


ROLES = ("work", "answer", "judge", "embedding")

# Each model has its own endpoint, whatever role calls it: <PREFIX>_API_KEY and <PREFIX>_BASE_URL in
# .env, and a wire protocol. GPT-6 Luna is called through the Responses API, Claude through Anthropic
# Messages. A model not listed here takes the role's variables (ANSWER_API_KEY, ...) and chat.
MODEL_CONNECTIONS = {
    "gpt-6-luna": ("GPT_6_LUNA", "responses"),
    "gpt-4o-mini": ("GPT_4O_MINI", "chat"),
    "text-embedding-3-small": ("TEXT_EMBEDDING_3_SMALL", "chat"),
    "claude-opus-5-5": ("CLAUDE_OPUS_5_5", "anthropic"),
}


def connection_settings(env_file=DEFAULT_ENV, *, models=None):
    values = {**dotenv_values(env_file), **os.environ}
    connections = {}
    for role in ROLES:
        model = (models or {}).get(role) or values.get(role.upper() + "_MODEL")
        prefix, protocol = MODEL_CONNECTIONS.get(model, (role.upper(), "chat"))
        url, key = values.get(prefix + "_BASE_URL"), values.get(prefix + "_API_KEY")
        if not all((url, key, model)):
            raise ValueError(f"{role}: set {prefix}_API_KEY, {prefix}_BASE_URL and a model")
        connections[role] = {
            "base_url": api_root(url),
            "key": key,
            "model": model,
            "credential_name": prefix + "_API_KEY",
        }
        if protocol != "chat":
            connections[role]["protocol"] = protocol
        if protocol == "responses":
            connections[role]["user_agent"] = "memory-tax/0.1"
        # Some routers reject default client User-Agents; set <PREFIX>_USER_AGENT for those.
        if values.get(prefix + "_USER_AGENT"):
            connections[role]["user_agent"] = values[prefix + "_USER_AGENT"]
        # Some routers route Responses requests by a per-conversation id in a header of their own;
        # <PREFIX>_SESSION_HEADER names it, and each client sends one id per role.
        if values.get(prefix + "_SESSION_HEADER"):
            connections[role]["session_header"] = values[prefix + "_SESSION_HEADER"]
        # Optional provider tier ("flex", say); it is part of the recorded payload.
        if values.get(prefix + "_SERVICE_TIER"):
            connections[role]["service_tier"] = values[prefix + "_SERVICE_TIER"]
    return connections


def public_connections(connections):
    return {
        role: {k: v for k, v in conn.items() if k != "key"}
        for role, conn in connections.items()
    }


class Client:
    def __init__(
        self,
        env_file=DEFAULT_ENV,
        *,
        models=None,
        reasoning_effort="none",
        timeout=90,
        transport=None,
    ):
        self.connections = connection_settings(env_file, models=models)
        self.reasoning_effort = reasoning_effort
        self.sessions = {role: "memory-tax-" + uuid.uuid4().hex for role in ROLES}
        self.local = threading.local()
        self.http_options = {"timeout": timeout, "transport": transport}
        self.http_clients = []
        self.http_lock = threading.Lock()

    @property
    def http(self):
        # A stalled connection must not exhaust other workers' connection pools.
        if not hasattr(self.local, "http"):
            self.local.http = httpx.Client(
                **self.http_options,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            )
            with self.http_lock:
                self.http_clients.append(self.local.http)
        return self.local.http

    @property
    def embedding_model(self):
        return self.connections["embedding"]["model"]

    @property
    def embedding_base_url(self):
        return self.connections["embedding"]["base_url"]

    @property
    def identity(self):
        return {
            "roles": public_connections(self.connections),
            "reasoning_effort": self.reasoning_effort,
        }

    def reset(self, scope=None):
        self.local.calls = []
        self.local.scope = Path(scope) if scope else None
        self.local.sequence = 0
        self.local.last = {}

    @property
    def calls(self):
        return getattr(self.local, "calls", [])

    def record(
        self, response, kind, model, *, phase="query", request=None, role="work"
    ):
        entry = {
            **usage_record(response, kind, model, role),
            "phase": phase,
            "base_url": self.connections[role]["base_url"],
            **getattr(self.local, "last", {}),
        }
        if request is not None:
            entry["request_sha256"] = digest(request)
        self.local.calls = self.calls + [entry]

    def post(self, endpoint, payload, *, role="work", attempts=4):
        if endpoint == "/embeddings":
            role = "embedding"
        if role == "embedding" and endpoint != "/embeddings":
            raise ValueError("embedding role only supports /embeddings")
        conn = self.connections[role]
        if payload.get("model") != conn["model"]:
            raise ValueError(f"{role}: request model differs from configured model")
        identity = {
            "role": role,
            "base_url": conn["base_url"],
            "endpoint": endpoint,
            "payload": payload,
        }
        if "protocol" in conn:
            identity["protocol"] = conn["protocol"]
        scope = getattr(self.local, "scope", None)
        sequence = getattr(self.local, "sequence", 0)
        self.local.sequence = sequence + 1
        directory = scope / "calls" / f"{sequence:05d}" if scope else None
        if directory:
            response_path = directory / "response.json"
            previous = verified(response_path, identity)
            if previous is not None:
                self.local.last = {
                    **previous["timing"],
                    "replayed": True,
                    "call_id": str(directory),
                    "base_url": conn["base_url"],
                }
                return previous["body"]
            intent = directory / "request.json"
            if intent.exists() and read(intent) != identity:
                raise ValueError("Resumed API request differs from recorded request")
            save(intent, identity)
        # HTTP 429/503 and in-band error bodies mean the request was not served; a saturated
        # upstream clears within minutes, so it is waited out on every path. Other errors keep
        # `attempts`.
        for attempt in range(max(attempts, SATURATION_ATTEMPTS)):
            started_at, started = now(), time.perf_counter()
            attempt_path = None
            if directory:
                old = sorted(directory.glob("attempt-*.json"))
                attempt_path = directory / f"attempt-{len(old) + 1:04d}.json"
                save(attempt_path, {"status": "running", "started_at": started_at})
            try:
                body, wire = self.send(conn, endpoint, payload, role)
                timing = {
                    "started_at": started_at,
                    "ended_at": now(),
                    "latency_s": time.perf_counter() - started,
                }
                if directory:
                    result = {"body": body, "timing": timing}
                    if wire is not None:
                        result["wire"] = wire
                    checkpoint(response_path, identity, result)
                    save(attempt_path, {"status": "succeeded", **timing})
                self.local.last = {
                    **timing,
                    "replayed": False,
                    "call_id": str(directory) if directory else None,
                    "base_url": conn["base_url"],
                }
                return body
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as error:
                status = (
                    error.response.status_code
                    if isinstance(error, httpx.HTTPStatusError)
                    else getattr(error, "status", None)
                )
                if attempt_path:
                    save(
                        attempt_path,
                        {
                            "status": "failed" if status else "unknown",
                            "http_status": status,
                            "error_type": type(error).__name__,
                            "started_at": started_at,
                            "ended_at": now(),
                            "latency_s": time.perf_counter() - started,
                            "billed_cost_usd": None,
                        },
                    )
                unserved = status in {429, 503, 529} or isinstance(error, UnservedError)
                transient = unserved or status in {408, 500, 502, 504} or isinstance(
                    error, httpx.TransportError
                )
                limit = SATURATION_ATTEMPTS if unserved else attempts
                if not transient or attempt >= limit - 1:
                    raise RuntimeError(
                        f"{role} request failed: {type(error).__name__}, status={status}"
                        + (f": {error}" if isinstance(error, StatusError) else "")
                    ) from None
                time.sleep(min(2**attempt, 60))

    def send(self, conn, endpoint, payload, role):
        """One HTTP attempt: the chat-shaped body, and the request and response on the wire
        when the provider speaks another protocol (None for chat completions)."""
        protocol = conn.get("protocol", "chat")
        if protocol == "chat":
            response = self.http.post(
                conn["base_url"] + endpoint,
                json=payload,
                headers={
                    "Authorization": "Bearer " + conn["key"],
                    **({"User-Agent": conn["user_agent"]} if "user_agent" in conn else {}),
                },
            )
            response.raise_for_status()
            body = response.json()
            if isinstance(body.get("error"), dict) and not (
                body.keys() & {"choices", "output", "data"}
            ):
                raise UnservedError(body["error"])
            return body, None
        if endpoint != "/chat/completions":
            raise ValueError(f"{protocol}: only chat completions are translated")
        if protocol == "responses":
            wire = responses_request(payload)
            response = self.http.post(
                conn["base_url"] + "/responses",
                json=wire,
                headers={
                    "Authorization": "Bearer " + conn["key"],
                    "User-Agent": conn["user_agent"],
                    **({conn["session_header"]: self.sessions[role]} if "session_header" in conn else {}),
                },
                timeout=REASONING_TIMEOUT,
            )
            response.raise_for_status()
            raw = response.json()
            if raw.get("status") == "failed" or (
                isinstance(raw.get("error"), dict) and not raw.get("output")
            ):
                raise UnservedError(raw.get("error") or {"message": "response failed"})
            return responses_to_chat(raw), {"request": wire, "response": raw}
        wire = anthropic_request(payload)
        try:
            message = self.anthropic_client(conn).messages.create(**wire)
        except anthropic.APIStatusError as error:
            raise StatusError(error.status_code, error.message) from None
        except anthropic.APIConnectionError as error:
            raise httpx.TransportError(type(error).__name__) from None
        raw = message.to_dict()
        return anthropic_to_chat(raw), {"request": wire, "response": raw}

    def anthropic_client(self, conn):
        if not hasattr(self.local, "anthropic"):
            # Retries stay in post(), which records every attempt.
            self.local.anthropic = anthropic.Anthropic(
                api_key=conn["key"], base_url=conn["base_url"].removesuffix("/v1"),
                max_retries=0, timeout=REASONING_TIMEOUT)
            with self.http_lock:
                self.http_clients.append(self.local.anthropic)
        return self.local.anthropic

    def generation_payload(self, payload, role="work"):
        payload = dict(payload)
        model = payload["model"]
        if "luna" in model or model.startswith("claude-"):
            # Neither takes sampling settings (Luna ignores them, Claude Opus 5.5 rejects them).
            payload.pop("temperature", None)
            payload.pop("top_p", None)
            if self.reasoning_effort is not None:
                payload["reasoning_effort"] = self.reasoning_effort
            if self.reasoning_effort not in (None, "none"):
                # The output limit covers reasoning too; a short answer limit would cut it off.
                for key in ("max_tokens", "max_completion_tokens"):
                    if payload.get(key) is not None and payload[key] < REASONING_OUTPUT_FLOOR:
                        payload[key] = REASONING_OUTPUT_FLOOR
        if "service_tier" in self.connections[role]:
            payload["service_tier"] = self.connections[role]["service_tier"]
        return payload

    def validate_generation_response(self, body, model):
        details = (
            body.get("usage", {}).get("completion_tokens_details")
            or body.get("usage", {}).get("output_tokens_details")
            or {}
        )
        if self.reasoning_effort == "none" and details.get("reasoning_tokens") not in (None, 0):
            raise ValueError(
                "Provider reported reasoning tokens despite reasoning_effort=none"
            )
        returned = body.get("model", "").removeprefix("openai/")
        requested = model.removeprefix("openai/")
        if returned and returned != requested and not returned.startswith(requested + "-"):
            raise ValueError("Provider returned an unexpected model")

    def chat(
        self,
        system,
        user,
        max_tokens=64,
        kind="generation",
        *,
        seed=0,
        response_format=None,
        role="work",
        transport_attempts=4,
        token_limit_parameter="max_completion_tokens",
    ):
        if token_limit_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported output-token limit parameter")
        model = self.connections[role]["model"]
        messages = [{"role": "system", "content": system}] if system is not None else []
        messages.append({"role": "user", "content": user})
        payload = {"model": model, "temperature": 0, "messages": messages}
        if seed is not None:
            payload["seed"] = seed
        if max_tokens is not None:
            payload[token_limit_parameter] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        payload = self.generation_payload(payload, role)
        body = self.post(
            "/chat/completions", payload, role=role, attempts=transport_attempts
        )
        self.record(body, kind, model, request=payload, role=role)
        self.validate_generation_response(body, model)
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty model response")
        if body["choices"][0].get("finish_reason") == "length":
            raise ValueError("Model response truncated by token limit")
        return content.strip()

    def embed(self, texts, *, phase="query"):
        model = requested = self.embedding_model
        payload = {"model": model, "input": texts}
        body = self.post("/embeddings", payload)
        self.record(
            body, "embedding", requested, phase=phase, request=payload, role="embedding"
        )
        data = sorted(body["data"], key=lambda x: x["index"])
        if [item["index"] for item in data] != list(range(len(texts))):
            raise ValueError("Embedding response indexes do not match inputs")
        vectors = [item["embedding"] for item in data]
        expected = (
            1536
            if requested.removeprefix("openai/") == "text-embedding-3-small"
            else None
        )
        if any(
            not isinstance(v, list)
            or not v
            or (expected is not None and len(v) != expected)
            or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in v)
            for v in vectors
        ):
            raise ValueError("Invalid embedding dimensions/values")
        return vectors

    def close(self):
        for client in self.http_clients:
            client.close()


def token_total(calls, kind, field=None):
    entries = [c for c in calls if c["kind"] == kind]
    fields = [field] if field else ["input_tokens", "output_tokens"]
    if any(c.get(f) is None for c in entries for f in fields):
        return None
    return sum(c[f] for c in entries for f in fields)
