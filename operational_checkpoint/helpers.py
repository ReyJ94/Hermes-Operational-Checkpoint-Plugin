from __future__ import annotations

import hashlib
import importlib
import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeGuard

from agent.model_metadata import get_model_context_length

SUMMARY_PREFIX: str = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)
LEGACY_SUMMARY_PREFIX: str = "[CONTEXT SUMMARY]:"

PLUGIN_ROOT: Path = Path(__file__).resolve().parents[1]
PLUGIN_CONFIG_PATH: Path = PLUGIN_ROOT / "operational_checkpoint.toml"
PACKAGED_PLUGIN_CONFIG_PATH: Path = Path(__file__).resolve().with_name(
    "operational_checkpoint.toml"
)
USAGE_SNAPSHOTS_PATH_SUFFIX: tuple[str, str] = (
    "operational_checkpoint",
    "usage_snapshots.json",
)
COMPACTION_STATES_PATH_SUFFIX: tuple[str, str] = (
    "operational_checkpoint",
    "compaction_states.json",
)


class LlmChoiceMessageLike(Protocol):
    content: object


class LlmChoiceLike(Protocol):
    message: object


class LlmResponseLike(Protocol):
    choices: object


@dataclass(frozen=True)
class PersistedCompactionState:
    compacted_messages: list[dict[str, object]]
    compression_count: int
    focus_topic: str | None
    raw_message_count: int
    summary: str
    tokens_after: int
    tokens_before: int
    updated_at: float
    checkpoint_id: str = ""
    previous_checkpoint_id: str = ""
    raw_cursor_message_count: int = 0
    raw_cursor_message_hash: str = ""


def require_attr(module: object, attr_name: str) -> object:
    if not hasattr(module, attr_name):
        error_message: str = f"Required attribute is unavailable: {attr_name}"
        raise TypeError(error_message)
    return getattr(module, attr_name)


def is_llm_response_like(candidate: object) -> TypeGuard[LlmResponseLike]:
    return hasattr(candidate, "choices")


def is_llm_choice_like(candidate: object) -> TypeGuard[LlmChoiceLike]:
    return hasattr(candidate, "message")


def is_llm_choice_message_like(candidate: object) -> TypeGuard[LlmChoiceMessageLike]:
    return hasattr(candidate, "content")


def load_plugin_root_config() -> dict[str, object]:
    raw_config: object
    config_path: Path
    for config_path in (PLUGIN_CONFIG_PATH, PACKAGED_PLUGIN_CONFIG_PATH):
        if not config_path.exists():
            continue
        with config_path.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
        return raw_config if isinstance(raw_config, dict) else {}
    return {}


def resolve_hermes_home(raw_path: object | None = None) -> Path:
    configured: str = string_or_empty(raw_path).strip()
    if not configured:
        configured = string_or_empty(os.getenv("HERMES_HOME")).strip()
    if not configured:
        return Path.home() / ".hermes"
    return Path(configured).expanduser()


def usage_snapshots_path(raw_hermes_home: object | None = None) -> Path:
    return resolve_hermes_home(raw_hermes_home).joinpath(*USAGE_SNAPSHOTS_PATH_SUFFIX)


def compaction_states_path(raw_hermes_home: object | None = None) -> Path:
    return resolve_hermes_home(raw_hermes_home).joinpath(*COMPACTION_STATES_PATH_SUFFIX)


def load_usage_snapshots(raw_hermes_home: object | None = None) -> dict[str, dict[str, object]]:
    path: Path = usage_snapshots_path(raw_hermes_home)
    if not path.exists():
        return {}
    try:
        raw_content: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(raw_content, dict):
        return {}

    snapshots: dict[str, dict[str, object]] = {}
    key: object
    value: object
    for key, value in raw_content.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        snapshot_payload: dict[str, object] = {}
        raw_key: object
        raw_value: object
        for raw_key, raw_value in value.items():
            snapshot_payload[str(raw_key)] = raw_value
        snapshots[key] = snapshot_payload
    return snapshots


def save_usage_snapshots(
    snapshots: Mapping[str, Mapping[str, object]],
    raw_hermes_home: object | None = None,
) -> None:
    path: Path = usage_snapshots_path(raw_hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized: dict[str, dict[str, object]] = {}
    session_id: str
    snapshot: Mapping[str, object]
    for session_id, snapshot in snapshots.items():
        if not isinstance(session_id, str):
            continue
        snapshot_payload: dict[str, object] = {}
        key: object
        value: object
        for key, value in snapshot.items():
            snapshot_payload[str(key)] = value
        serialized[session_id] = snapshot_payload
    temp_path: Path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(serialized, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _coerce_message_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    coerced: dict[str, object] = {}
    key: object
    raw_value: object
    for key, raw_value in value.items():
        coerced[str(key)] = raw_value
    return coerced


def _message_hash(message: dict[str, object]) -> str:
    payload: str = json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cursor_hash_for_messages(messages: list[dict[str, object]], raw_message_count: int) -> str:
    if raw_message_count <= 0 or raw_message_count > len(messages):
        return ""
    return _message_hash(messages[raw_message_count - 1])


def load_compaction_states(
    raw_hermes_home: object | None = None,
) -> dict[str, PersistedCompactionState]:
    path: Path = compaction_states_path(raw_hermes_home)
    if not path.exists():
        return {}
    try:
        raw_content: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(raw_content, dict):
        return {}

    states: dict[str, PersistedCompactionState] = {}
    session_id: object
    state_value: object
    for session_id, state_value in raw_content.items():
        if not isinstance(session_id, str) or not isinstance(state_value, dict):
            continue
        raw_messages: object = state_value.get("compacted_messages")
        compacted_messages: list[dict[str, object]] = []
        if isinstance(raw_messages, list):
            raw_message: object
            for raw_message in raw_messages:
                message_dict: dict[str, object] | None = _coerce_message_dict(raw_message)
                if message_dict is not None:
                    compacted_messages.append(message_dict)
        if not compacted_messages:
            continue

        focus_topic_text: str = string_or_empty(state_value.get("focus_topic")).strip()
        updated_at_raw: object = state_value.get("updated_at")
        updated_at: float = (
            float(updated_at_raw)
            if isinstance(updated_at_raw, (float, int)) and not isinstance(updated_at_raw, bool)
            else 0.0
        )
        raw_cursor_count: int = as_positive_int(
            state_value.get("raw_cursor_message_count"),
            as_positive_int(state_value.get("raw_message_count"), 0),
        )
        raw_cursor_hash: str = string_or_empty(
            state_value.get("raw_cursor_message_hash")
        ).strip()
        if not raw_cursor_hash:
            raw_cursor_hash = _cursor_hash_for_messages(
                compacted_messages,
                min(raw_cursor_count, len(compacted_messages)),
            )
        states[session_id] = PersistedCompactionState(
            compacted_messages=compacted_messages,
            compression_count=as_positive_int(state_value.get("compression_count"), 0),
            focus_topic=focus_topic_text or None,
            raw_message_count=as_positive_int(state_value.get("raw_message_count"), 0),
            summary=string_or_empty(state_value.get("summary")).strip(),
            tokens_after=as_positive_int(state_value.get("tokens_after"), 0),
            tokens_before=as_positive_int(state_value.get("tokens_before"), 0),
            updated_at=updated_at,
            checkpoint_id=string_or_empty(state_value.get("checkpoint_id")).strip(),
            previous_checkpoint_id=string_or_empty(
                state_value.get("previous_checkpoint_id")
            ).strip(),
            raw_cursor_message_count=raw_cursor_count,
            raw_cursor_message_hash=raw_cursor_hash,
        )
    return states


def save_compaction_states(
    states: Mapping[str, PersistedCompactionState],
    raw_hermes_home: object | None = None,
) -> None:
    path: Path = compaction_states_path(raw_hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized: dict[str, dict[str, object]] = {}
    session_id: str
    state: PersistedCompactionState
    for session_id, state in states.items():
        if not isinstance(session_id, str):
            continue
        compacted_messages: list[dict[str, object]] = []
        message: dict[str, object]
        for message in state.compacted_messages:
            if not isinstance(message, dict):
                continue
            message_payload: dict[str, object] = {}
            key: object
            value: object
            for key, value in message.items():
                message_payload[str(key)] = value
            compacted_messages.append(message_payload)
        serialized[session_id] = {
            "checkpoint_id": state.checkpoint_id,
            "compacted_messages": compacted_messages,
            "compression_count": int(state.compression_count),
            "focus_topic": state.focus_topic,
            "previous_checkpoint_id": state.previous_checkpoint_id,
            "raw_cursor_message_count": int(
                state.raw_cursor_message_count or state.raw_message_count
            ),
            "raw_cursor_message_hash": state.raw_cursor_message_hash,
            "raw_message_count": int(state.raw_message_count),
            "summary": state.summary,
            "tokens_after": int(state.tokens_after),
            "tokens_before": int(state.tokens_before),
            "updated_at": float(state.updated_at),
        }
    temp_path: Path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(serialized, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def load_config() -> dict[str, object]:
    load_config_attr: object = require_attr(
        importlib.import_module("hermes_cli.config"),
        "load_config",
    )
    if not callable(load_config_attr):
        raise TypeError("hermes_cli.config.load_config is unavailable")
    raw_config: object = load_config_attr()
    return raw_config if isinstance(raw_config, dict) else {}


def call_llm(
    *,
    task: str | None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    messages: list[dict[str, str]],
    max_tokens: int | None,
    extra_body: dict[str, object] | None = None,
    main_runtime: dict[str, object] | None = None,
) -> object:
    call_llm_attr: object = require_attr(
        importlib.import_module("agent.auxiliary_client"),
        "call_llm",
    )
    if not callable(call_llm_attr):
        raise TypeError("agent.auxiliary_client.call_llm is unavailable")
    call_kwargs: dict[str, object] = {
        "task": task,
        "messages": messages,
        "extra_body": extra_body,
    }
    if provider:
        call_kwargs["provider"] = provider
    if model:
        call_kwargs["model"] = model
    if base_url:
        call_kwargs["base_url"] = base_url
    if api_key:
        call_kwargs["api_key"] = api_key
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    if main_runtime is not None:
        call_kwargs["main_runtime"] = main_runtime
    return call_llm_attr(**call_kwargs)


def extract_choice_content(response: object) -> str:
    choices: object
    first_choice: object
    message: object
    content: object

    if not is_llm_response_like(response):
        return ""

    choices = response.choices
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not is_llm_choice_like(first_choice):
        return ""

    message = first_choice.message
    if not is_llm_choice_message_like(message):
        return ""

    content = message.content
    if isinstance(content, str):
        return content
    return str(content) if content else ""


def as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(raw_key): raw_value for raw_key, raw_value in value.items()}
    return {}


def as_positive_int(value: object, default: int) -> int:
    parsed: int
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value if value > 0 else default
    if isinstance(value, str) and value.strip():
        try:
            parsed = int(value)
        except ValueError:
            return default
        return parsed if parsed > 0 else default
    return default


def require_positive_int(value: object, key_name: str) -> int:
    parsed: int
    error_message: str
    if isinstance(value, bool):
        error_message = (
            f"operational_checkpoint.{key_name} must be a positive integer"
        )
        raise TypeError(error_message)
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip():
        parsed = int(value)
    else:
        error_message = f"operational_checkpoint.{key_name} is required"
        raise TypeError(error_message)
    if parsed <= 0:
        error_message = (
            f"operational_checkpoint.{key_name} must be greater than zero"
        )
        raise ValueError(error_message)
    return parsed


def require_non_negative_int(value: object, key_name: str) -> int:
    parsed: int
    error_message: str
    if isinstance(value, bool):
        error_message = (
            f"operational_checkpoint.{key_name} must be a non-negative integer"
        )
        raise TypeError(error_message)
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip():
        parsed = int(value)
    else:
        error_message = f"operational_checkpoint.{key_name} is required"
        raise TypeError(error_message)
    if parsed < 0:
        error_message = (
            f"operational_checkpoint.{key_name} must be zero or greater"
        )
        raise ValueError(error_message)
    return parsed


def require_string(value: object, key_name: str) -> str:
    parsed: str = string_or_empty(value).strip()
    if not parsed:
        error_message: str = f"operational_checkpoint.{key_name} is required"
        raise TypeError(error_message)
    return parsed



def require_fraction(value: object, key_name: str) -> float:
    if isinstance(value, bool):
        error_message = f"operational_checkpoint.{key_name} must be a number between 0 and 1"
        raise TypeError(error_message)
    if isinstance(value, (float, int)):
        parsed = float(value)
    elif isinstance(value, str) and value.strip():
        parsed = float(value)
    else:
        error_message = f"operational_checkpoint.{key_name} is required"
        raise TypeError(error_message)
    if parsed <= 0 or parsed >= 1:
        error_message = f"operational_checkpoint.{key_name} must be greater than 0 and less than 1"
        raise ValueError(error_message)
    return parsed


def string_or_empty(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def estimate_tokens(text: str | None) -> int:
    normalized: str
    if text is None:
        return 0

    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, (len(normalized) + 3) // 4)


def normalize_reasoning_effort(value: object) -> str:
    raw: str = string_or_empty(value).strip().lower()
    if raw in {"none", "minimal", "low", "medium", "high", "xhigh"}:
        return raw
    return ""


def load_operational_checkpoint_config() -> dict[str, object]:
    config: dict[str, object] = load_config()
    plugin_root_config: dict[str, object] = load_plugin_root_config()
    plugin_defaults: dict[str, object] = as_mapping(plugin_root_config.get("defaults"))
    hermes_overrides: dict[str, object] = as_mapping(
        config.get("operational_checkpoint")
    )
    return {**plugin_defaults, **hermes_overrides}


def load_operational_checkpoint_cli_config() -> dict[str, object]:
    config: dict[str, object] = load_config()
    plugin_root_config: dict[str, object] = load_plugin_root_config()
    plugin_cli_defaults: dict[str, object] = as_mapping(plugin_root_config.get("cli"))
    hermes_overrides: dict[str, object] = as_mapping(config.get("operational_checkpoint"))
    hermes_cli_overrides: dict[str, object] = as_mapping(hermes_overrides.get("cli"))
    return {**plugin_cli_defaults, **hermes_cli_overrides}


def load_runtime_defaults() -> dict[str, object]:
    config: dict[str, object] = load_config()
    plugin_root_config: dict[str, object] = load_plugin_root_config()
    plugin_root_defaults: dict[str, object] = as_mapping(
        plugin_root_config.get("defaults")
    )
    plugin_cfg: dict[str, object] = load_operational_checkpoint_config()
    model_cfg: dict[str, object] = as_mapping(config.get("model"))

    runtime_model: str = string_or_empty(model_cfg.get("default")).strip()
    runtime_provider: str = string_or_empty(model_cfg.get("provider")).strip()
    runtime_base_url: str = string_or_empty(model_cfg.get("base_url"))
    context_limit_tokens: int = get_model_context_length(
        model=runtime_model,
        base_url=runtime_base_url,
        api_key=string_or_empty(model_cfg.get("api_key")),
        config_context_length=as_positive_int(model_cfg.get("context_length"), 0) or None,
        provider=runtime_provider,
    )

    compression_cfg: dict[str, object] = as_mapping(config.get("compression"))
    threshold_percent_value: object = plugin_root_defaults.get(
        "compaction_threshold_percent"
    )
    if threshold_percent_value is None:
        threshold_percent_value = plugin_root_defaults.get("threshold_percent")
    if threshold_percent_value is None:
        threshold_percent_value = compression_cfg.get("threshold")
    if threshold_percent_value is None and plugin_cfg.get("auto_compact_at_tokens") is not None:
        # Legacy compatibility for older configs/tests. New configs should set
        # the repo-owned compaction_threshold_percent and let provider+model
        # metadata own the context window. Per-user Hermes overrides are not the
        # source of truth for Operational Checkpoint's agreed checkpoint policy.
        threshold_percent = require_positive_int(
            plugin_cfg.get("auto_compact_at_tokens"),
            "auto_compact_at_tokens",
        ) / context_limit_tokens
        if threshold_percent <= 0 or threshold_percent >= 1:
            raise ValueError(
                "operational_checkpoint.auto_compact_at_tokens must resolve to "
                "a fraction greater than 0 and less than the provider/model context window"
            )
    else:
        threshold_percent = require_fraction(
            threshold_percent_value,
            "compaction_threshold_percent",
        )
    auto_compact_at_tokens: int = max(1, int(context_limit_tokens * threshold_percent))

    tail_preserve_tokens: int = require_non_negative_int(
        plugin_cfg.get("tail_preserve_tokens"),
        "tail_preserve_tokens",
    )
    if tail_preserve_tokens >= auto_compact_at_tokens:
        raise ValueError(
            "operational_checkpoint.tail_preserve_tokens must be less than "
            "the configured compaction threshold"
        )

    return {
        "auto_compact_at_tokens": auto_compact_at_tokens,
        "base_url": runtime_base_url,
        "compaction_threshold_percent": threshold_percent,
        "config_context_length": None,
        "context_limit_tokens": context_limit_tokens,
        "head_preserve_messages": require_non_negative_int(
            plugin_cfg.get("head_preserve_messages"),
            "head_preserve_messages",
        ),
        "minimum_tail_messages": require_non_negative_int(
            plugin_cfg.get("minimum_tail_messages"),
            "minimum_tail_messages",
        ),
        "model": require_string(plugin_root_defaults.get("model"), "model"),
        "provider": runtime_provider,
        "reasoning_effort": normalize_reasoning_effort(
            plugin_root_defaults.get("reasoning_effort")
        ),
        "summary_retry_attempts": require_positive_int(
            plugin_cfg.get("summary_retry_attempts"),
            "summary_retry_attempts",
        ),
        "tail_preserve_tokens": tail_preserve_tokens,
    }


def strip_summary_prefix(summary_text: str) -> str:
    text: str = (summary_text or "").strip()
    prefix: str
    for prefix in (SUMMARY_PREFIX, LEGACY_SUMMARY_PREFIX):
        if text.startswith(prefix):
            return text[len(prefix) :].lstrip()
    return text
