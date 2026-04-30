from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeGuard
from weakref import WeakKeyDictionary

from agent.model_metadata import (
    estimate_request_tokens_rough,
)

from .helpers import (
    PersistedCompactionState,
    as_positive_int,
    load_compaction_states,
    load_operational_checkpoint_cli_config,
    save_compaction_states,
    string_or_empty,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import ModuleType

_COMPRESS_CONTEXT_RESULT_LEN: int = 2
_DEFERRED_INSTALL_POLL_INTERVAL_SECONDS: float = 0.05
_DEFERRED_INSTALL_TIMEOUT_SECONDS: float = 5.0
_TRIGGER_STATE: threading.local = threading.local()
_SIDECAR_INSTALL_LOCK: threading.Lock = threading.Lock()
_SIDECAR_INSTALL_SCHEDULED: bool = False
logger: logging.Logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CLIRuntimeState:
    last_compaction_artifact: dict[str, object] | None = None


@dataclass(slots=True)
class AgentRuntimeState:
    cli: object | None = None
    compaction_state: PersistedCompactionState | None = None
    hydrated_cursor: int | None = None
    last_compaction_artifact: dict[str, object] | None = None


_CLI_RUNTIME_STATE: WeakKeyDictionary[object, CLIRuntimeState] = WeakKeyDictionary()
_AGENT_RUNTIME_STATE: WeakKeyDictionary[object, AgentRuntimeState] = WeakKeyDictionary()


class CompactionCallbackEngineLike(Protocol):
    name: str
    compression_count: int
    context_length: int
    hermes_home: object
    last_checkpoint_summary: str
    last_completion_tokens: int
    last_prompt_tokens: int
    last_total_tokens: int
    threshold_tokens: int

    def compress(
        self,
        messages: list[dict[str, object]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> list[dict[str, object]]: ...

    def update_from_response(self, usage: dict[str, object]) -> object: ...

    def set_compaction_callback(
        self,
        callback: Callable[[dict[str, str | None]], None] | None,
    ) -> None: ...

    def bind_session(
        self,
        *,
        session_id: str,
        hermes_home: str | None = None,
        parent_session_id: str | None = None,
    ) -> None: ...

    def restore_usage_snapshot(self) -> bool: ...

    def persist_usage_snapshot(self) -> None: ...


class CLIBridgeTarget(Protocol):
    session_id: str

    def _init_agent(self, *args: object, **kwargs: object) -> bool: ...

    def _manual_compress(self, cmd_original: str = "") -> None: ...


class AgentBridgeTarget(Protocol):
    def _compress_context(self, *args: object, **kwargs: object) -> tuple[object, object]: ...


def _cli_runtime_state(cli: object) -> CLIRuntimeState:
    state: CLIRuntimeState | None = _CLI_RUNTIME_STATE.get(cli)
    if state is None:
        state = CLIRuntimeState()
        _CLI_RUNTIME_STATE[cli] = state
    return state


def _agent_runtime_state(agent: object) -> AgentRuntimeState:
    state: AgentRuntimeState | None = _AGENT_RUNTIME_STATE.get(agent)
    if state is None:
        state = AgentRuntimeState()
        _AGENT_RUNTIME_STATE[agent] = state
    return state


@contextmanager
def compaction_trigger_scope(
    trigger: Literal["auto", "manual"],
) -> Generator[None, None, None]:
    previous_trigger: str = string_or_empty(getattr(_TRIGGER_STATE, "trigger", "")).strip()
    _TRIGGER_STATE.trigger = trigger
    try:
        yield
    finally:
        if previous_trigger:
            _TRIGGER_STATE.trigger = previous_trigger
        elif hasattr(_TRIGGER_STATE, "trigger"):
            delattr(_TRIGGER_STATE, "trigger")


@contextmanager
def compaction_state_scope(state: dict[str, object]) -> Generator[None, None, None]:
    previous_state: object = getattr(_TRIGGER_STATE, "state", None)
    _TRIGGER_STATE.state = state
    try:
        yield
    finally:
        if previous_state is None:
            if hasattr(_TRIGGER_STATE, "state"):
                delattr(_TRIGGER_STATE, "state")
        else:
            _TRIGGER_STATE.state = previous_state


@contextmanager
def swallow_compaction_preview() -> Generator[None, None, None]:
    previous_value: object = getattr(_TRIGGER_STATE, "suppress_preview", None)
    _TRIGGER_STATE.suppress_preview = True
    try:
        yield
    finally:
        if previous_value is None:
            if hasattr(_TRIGGER_STATE, "suppress_preview"):
                delattr(_TRIGGER_STATE, "suppress_preview")
        else:
            _TRIGGER_STATE.suppress_preview = previous_value


def current_compaction_trigger() -> Literal["auto", "manual"]:
    raw_trigger: str = string_or_empty(getattr(_TRIGGER_STATE, "trigger", "auto")).strip().lower()
    return "manual" if raw_trigger == "manual" else "auto"


def current_compaction_state() -> dict[str, object] | None:
    raw_state: object = getattr(_TRIGGER_STATE, "state", None)
    return raw_state if isinstance(raw_state, dict) else None


def should_suppress_preview() -> bool:
    return bool(getattr(_TRIGGER_STATE, "suppress_preview", False))


def is_compaction_callback_engine(
    candidate: object,
) -> TypeGuard[CompactionCallbackEngineLike]:
    return (
        hasattr(candidate, "name")
        and string_or_empty(getattr(candidate, "name", "")).strip()
        == "operational_checkpoint"
        and callable(getattr(candidate, "set_compaction_callback", None))
    )


def is_cli_bridge_target(candidate: object) -> TypeGuard[CLIBridgeTarget]:
    return (
        hasattr(candidate, "session_id")
        and callable(getattr(candidate, "_init_agent", None))
        and callable(getattr(candidate, "_manual_compress", None))
    )


def _load_cli_bridge_config() -> dict[str, object]:
    return load_operational_checkpoint_cli_config()


def _should_emit_status() -> bool:
    config: dict[str, object] = _load_cli_bridge_config()
    raw_value: object = config.get("emit_compaction_status")
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        return raw_value.strip().lower() not in {"", "0", "false", "no", "off"}
    return True


def _should_show_summary_preview() -> bool:
    config: dict[str, object] = _load_cli_bridge_config()
    raw_value: object = config.get("show_summary_preview")
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _summary_preview_chars() -> int:
    config: dict[str, object] = _load_cli_bridge_config()
    return as_positive_int(config.get("summary_preview_chars"), 160)


def _safe_emit(agent: object, line: str) -> None:
    safe_print: object = getattr(agent, "_safe_print", None)
    if callable(safe_print):
        safe_print(line)
        return
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()

def _bind_engine_session_state(agent: object, engine: CompactionCallbackEngineLike) -> None:
    engine.bind_session(
        session_id=string_or_empty(getattr(agent, "session_id", "")),
        hermes_home=string_or_empty(getattr(agent, "hermes_home", "")) or None,
        parent_session_id=None,
    )
    engine.restore_usage_snapshot()


def _persist_engine_session_state(engine: CompactionCallbackEngineLike) -> None:
    engine.persist_usage_snapshot()


def _find_tui_server_for_agent(agent: object) -> ModuleType | None:
    module: ModuleType
    for module in list(sys.modules.values()):
        sessions: object = getattr(module, "_sessions", None)
        emit: object = getattr(module, "_emit", None)
        session_info: object = getattr(module, "_session_info", None)
        if not isinstance(sessions, dict) or not callable(emit) or not callable(session_info):
            continue
        session: object
        for session in sessions.values():
            if isinstance(session, dict) and session.get("agent") is agent:
                return module
    return None


def _emit_tui_usage_update(agent: object) -> None:
    """Refresh the TUI's existing session.info usage snapshot after token changes."""
    module: ModuleType | None = _find_tui_server_for_agent(agent)
    if module is None:
        return
    sessions: object = getattr(module, "_sessions", None)
    emit: object = getattr(module, "_emit", None)
    session_info: object = getattr(module, "_session_info", None)
    if not isinstance(sessions, dict) or not callable(emit) or not callable(session_info):
        return
    sid: str
    session: object
    for sid, session in sessions.items():
        if not isinstance(sid, str) or not isinstance(session, dict):
            continue
        if session.get("agent") is not agent:
            continue
        try:
            info_payload: object = session_info(agent)
            if isinstance(info_payload, dict):
                emit("session.info", sid, info_payload)
        except Exception:
            logger.debug("Operational Checkpoint TUI usage update failed", exc_info=True)
        return


def _install_usage_update_hook(agent: object, engine: CompactionCallbackEngineLike) -> None:
    installed: object = getattr(engine, "_operational_checkpoint_usage_hook_installed", False)
    if installed:
        return
    original_update: object = getattr(engine, "update_from_response", None)
    if not callable(original_update):
        return

    def wrapped_update_from_response(usage: dict[str, object]) -> object:
        result: object = original_update(usage)
        _emit_tui_usage_update(agent)
        return result

    object.__setattr__(engine, "update_from_response", wrapped_update_from_response)
    object.__setattr__(engine, "_operational_checkpoint_usage_hook_installed", True)


def _bind_agent_runtime_hooks(agent: object) -> None:
    engine: object = getattr(agent, "context_compressor", None)
    if not is_compaction_callback_engine(engine):
        return

    def callback(payload: dict[str, str | None]) -> None:
        _record_compaction_artifact(agent=agent, payload=payload)

    engine.set_compaction_callback(callback)
    _install_usage_update_hook(agent, engine)
    if string_or_empty(getattr(agent, "platform", "")).strip().lower() == "tui":
        _bind_engine_session_state(agent, engine)
    _install_tui_session_compress_response_hook()


def _active_system_prompt(agent: object, fallback: str) -> str:
    cached_system_prompt: str = string_or_empty(
        getattr(agent, "_cached_system_prompt", "")
    )
    return cached_system_prompt or fallback


def _estimate_request_tokens(
    *,
    agent: object,
    messages: list[dict[str, object]],
    system_prompt: str,
) -> int:
    return estimate_request_tokens_rough(
        messages,
        system_prompt=system_prompt,
        tools=getattr(agent, "tools", None),
    )


def _current_request_tokens(
    *,
    agent: object,
    engine: object,
    messages: list[dict[str, object]],
    system_prompt: str,
) -> int:
    estimated_request_tokens: int = _estimate_request_tokens(
        agent=agent,
        messages=messages,
        system_prompt=system_prompt,
    )
    last_prompt_tokens: object = getattr(engine, "last_prompt_tokens", None)
    last_completion_tokens: object = getattr(engine, "last_completion_tokens", None)
    tracked_request_tokens: int = 0
    if (
        isinstance(last_prompt_tokens, int)
        and last_prompt_tokens >= 0
        and isinstance(last_completion_tokens, int)
        and last_completion_tokens >= 0
    ):
        tracked_request_tokens = last_prompt_tokens + last_completion_tokens
    return max(estimated_request_tokens, tracked_request_tokens)


def _copy_message_list(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    message: dict[str, object]
    for message in messages:
        copied.append({str(key): value for key, value in message.items()})
    return copied


def latest_compaction_artifact_for_agent(agent: object) -> dict[str, object] | None:
    artifact: dict[str, object] | None = _agent_runtime_state(agent).last_compaction_artifact
    return dict(artifact) if artifact is not None else None


def latest_compaction_artifact_for_cli(cli: object) -> dict[str, object] | None:
    artifact: dict[str, object] | None = _cli_runtime_state(cli).last_compaction_artifact
    return dict(artifact) if artifact is not None else None


def hydrated_cursor_for_agent(agent: object) -> int | None:
    return _agent_runtime_state(agent).hydrated_cursor


def _active_compaction_state(agent: object) -> PersistedCompactionState | None:
    return _agent_runtime_state(agent).compaction_state


def _set_active_compaction_state(
    *,
    agent: object,
    record: PersistedCompactionState | None,
) -> None:
    runtime_state: AgentRuntimeState = _agent_runtime_state(agent)
    runtime_state.compaction_state = record
    if record is None:
        runtime_state.hydrated_cursor = None
        return
    runtime_state.hydrated_cursor = len(record.compacted_messages)


def _raw_message_count_for_hydrated_messages(
    agent: object,
    messages: list[dict[str, object]],
) -> int:
    record: PersistedCompactionState | None = _active_compaction_state(agent)
    if record is None:
        return len(messages)
    compacted_base_count: int = len(record.compacted_messages)
    suffix_count: int = max(0, len(messages) - compacted_base_count)
    return record.raw_message_count + suffix_count


def _persist_compaction_state(
    *,
    agent: object,
    engine: CompactionCallbackEngineLike,
    compressed_messages: list[dict[str, object]],
    focus_topic: str | None,
    raw_message_count: int,
    tokens_after: int,
    tokens_before: int,
) -> PersistedCompactionState | None:
    session_id: str = string_or_empty(getattr(agent, "session_id", "")).strip()
    if not session_id:
        return None

    hermes_home: object = getattr(engine, "hermes_home", None)
    states: dict[str, PersistedCompactionState] = load_compaction_states(hermes_home)
    summary: str = string_or_empty(getattr(engine, "last_checkpoint_summary", "")).strip()
    record = PersistedCompactionState(
        compacted_messages=_copy_message_list(compressed_messages),
        compression_count=int(getattr(engine, "compression_count", 0) or 0),
        focus_topic=focus_topic,
        raw_message_count=max(0, raw_message_count),
        summary=summary,
        tokens_after=max(0, tokens_after),
        tokens_before=max(0, tokens_before),
        updated_at=time.time(),
    )
    states[session_id] = record
    save_compaction_states(states, hermes_home)
    _set_active_compaction_state(agent=agent, record=record)
    return record


def _load_persisted_compaction_state(
    *,
    session_id: str,
    hermes_home: object,
) -> PersistedCompactionState | None:
    states: dict[str, PersistedCompactionState] = load_compaction_states(hermes_home)
    return states.get(session_id)


def _hydrate_messages_from_record(
    *,
    raw_messages: list[dict[str, object]],
    record: PersistedCompactionState,
) -> list[dict[str, object]]:
    raw_tail_start: int = min(record.raw_message_count, len(raw_messages))
    hydrated_messages: list[dict[str, object]] = _copy_message_list(record.compacted_messages)
    hydrated_messages.extend(_copy_message_list(raw_messages[raw_tail_start:]))
    return hydrated_messages


def _raw_messages_for_session_log(
    *,
    agent: object,
    messages: list[dict[str, object]],
    record: PersistedCompactionState,
) -> list[dict[str, object]]:
    session_db: object = getattr(agent, "_session_db", None)
    session_id: str = string_or_empty(getattr(agent, "session_id", "")).strip()
    if session_db is None or not session_id:
        return _copy_message_list(messages)

    get_messages: object = getattr(session_db, "get_messages_as_conversation", None)
    if not callable(get_messages):
        return _copy_message_list(messages)

    restored: object = get_messages(session_id)
    if not isinstance(restored, list):
        return _copy_message_list(messages)

    raw_prefix: list[dict[str, object]] = []
    restored_message: object
    for restored_message in restored[: record.raw_message_count]:
        if isinstance(restored_message, dict):
            raw_prefix.append(
                {str(key): value for key, value in restored_message.items()}
            )
    compacted_base_count: int = len(record.compacted_messages)
    raw_suffix: list[dict[str, object]] = _copy_message_list(messages[compacted_base_count:])
    return raw_prefix + raw_suffix

def _activate_hydrated_session_state(
    cli: object,
    agent: object,
    engine: CompactionCallbackEngineLike,
) -> None:
    session_id: str = string_or_empty(getattr(agent, "session_id", "")).strip()
    if not session_id:
        return
    raw_history: object = getattr(cli, "conversation_history", None)
    if not isinstance(raw_history, list):
        return

    record: PersistedCompactionState | None = _load_persisted_compaction_state(
        session_id=session_id,
        hermes_home=getattr(engine, "hermes_home", None),
    )
    if record is None:
        _set_active_compaction_state(agent=agent, record=None)
        return

    hydrated_history: list[dict[str, object]] = _hydrate_messages_from_record(
        raw_messages=raw_history,
        record=record,
    )
    object.__setattr__(cli, "conversation_history", hydrated_history)
    object.__setattr__(agent, "_session_messages", hydrated_history)
    _set_active_compaction_state(agent=agent, record=record)
    object.__setattr__(
        agent,
        "_last_flushed_db_idx",
        record.raw_message_count + max(0, len(hydrated_history) - len(record.compacted_messages)),
    )

    hydrated_tokens: int = _estimate_request_tokens(
        agent=agent,
        messages=hydrated_history,
        system_prompt=_active_system_prompt(agent, ""),
    )
    engine.last_prompt_tokens = hydrated_tokens
    engine.last_completion_tokens = 0
    engine.last_total_tokens = hydrated_tokens
    _persist_engine_session_state(engine)


def _hydrate_cli_history_from_plugin_state(
    cli: object,
    *,
    hermes_home: object | None = None,
) -> None:
    session_id: str = string_or_empty(getattr(cli, "session_id", "")).strip()
    raw_history: object = getattr(cli, "conversation_history", None)
    if not session_id or not isinstance(raw_history, list):
        return

    record: PersistedCompactionState | None = _load_persisted_compaction_state(
        session_id=session_id,
        hermes_home=hermes_home,
    )
    if record is None:
        return

    hydrated_history: list[dict[str, object]] = _hydrate_messages_from_record(
        raw_messages=raw_history,
        record=record,
    )
    object.__setattr__(cli, "conversation_history", hydrated_history)
def _reset_file_dedup(task_id: str) -> None:
    try:
        from tools.file_tools import reset_file_dedup

        reset_file_dedup(task_id)
    except Exception:
        logger.debug("Operational Checkpoint could not reset file dedup", exc_info=True)


def _perform_plugin_owned_compaction(
    agent: object,
    *,
    messages: list[dict[str, object]],
    system_message: str,
    approx_tokens: int | None,
    task_id: str,
    focus_topic: str | None,
) -> tuple[list[dict[str, object]], str]:
    engine: object = getattr(agent, "context_compressor", None)
    if not is_compaction_callback_engine(engine):
        raise TypeError("Operational Checkpoint engine is unavailable")
    del approx_tokens

    active_system_prompt: str = _active_system_prompt(agent, system_message)
    current_request_tokens: int = _current_request_tokens(
        agent=agent,
        engine=engine,
        messages=messages,
        system_prompt=active_system_prompt,
    )
    raw_message_count: int = _raw_message_count_for_hydrated_messages(agent, messages)
    logger.info(
        "operational checkpoint compaction started: session=%s messages=%d tokens=~%s focus=%r",
        string_or_empty(getattr(agent, "session_id", "")).strip() or "none",
        raw_message_count,
        f"{current_request_tokens:,}",
        focus_topic,
    )

    flush_memories: object = getattr(agent, "flush_memories", None)
    if callable(flush_memories):
        flush_memories(messages, min_turns=0)

    memory_manager: object = getattr(agent, "_memory_manager", None)
    if memory_manager is not None:
        on_pre_compress: object = getattr(memory_manager, "on_pre_compress", None)
        if callable(on_pre_compress):
            try:
                on_pre_compress(messages)
            except Exception:
                logger.debug(
                    "Operational Checkpoint memory pre-compress hook failed",
                    exc_info=True,
                )

    compressed: list[dict[str, object]] = list(
        engine.compress(
            messages,
            current_tokens=current_request_tokens,
            focus_topic=focus_topic,
        )
    )

    todo_store: object = getattr(agent, "_todo_store", None)
    format_for_injection: object = getattr(todo_store, "format_for_injection", None)
    if callable(format_for_injection):
        todo_snapshot: str = string_or_empty(format_for_injection())
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

    invalidate_system_prompt: object = getattr(agent, "_invalidate_system_prompt", None)
    if callable(invalidate_system_prompt):
        invalidate_system_prompt()

    new_system_prompt: str = system_message
    build_system_prompt: object = getattr(agent, "_build_system_prompt", None)
    if callable(build_system_prompt):
        new_system_prompt = string_or_empty(build_system_prompt(system_message))
        object.__setattr__(agent, "_cached_system_prompt", new_system_prompt)

    compacted_tokens: int = _estimate_request_tokens(
        agent=agent,
        messages=compressed,
        system_prompt=new_system_prompt,
    )
    engine.last_prompt_tokens = compacted_tokens
    engine.last_completion_tokens = 0
    engine.last_total_tokens = compacted_tokens
    _persist_engine_session_state(engine)
    _emit_tui_usage_update(agent)
    record: PersistedCompactionState | None = _persist_compaction_state(
        agent=agent,
        engine=engine,
        compressed_messages=compressed,
        focus_topic=focus_topic,
        raw_message_count=raw_message_count,
        tokens_after=compacted_tokens,
        tokens_before=current_request_tokens,
    )
    if record is not None:
        object.__setattr__(agent, "_session_messages", list(compressed))

    _reset_file_dedup(task_id)

    logger.info(
        "operational checkpoint compaction done: session=%s messages=%d tokens=~%s",
        string_or_empty(getattr(agent, "session_id", "")).strip() or "none",
        len(compressed),
        f"{compacted_tokens:,}",
    )
    return compressed, new_system_prompt


def _fmt_tokens(value: int | None) -> str:
    if isinstance(value, int) and value >= 0:
        return f"~{value:,}"
    return "unknown"


def _tui_compaction_summary(artifact: dict[str, object]) -> dict[str, object]:
    tokens_before_raw: object = artifact.get("tokens_before")
    tokens_after_raw: object = artifact.get("tokens_after")
    compression_count_raw: object = artifact.get("compression_count")
    tokens_before: int | None = (
        tokens_before_raw if isinstance(tokens_before_raw, int) else None
    )
    tokens_after: int | None = (
        tokens_after_raw if isinstance(tokens_after_raw, int) else None
    )
    compression_count: int = (
        compression_count_raw if isinstance(compression_count_raw, int) else 0
    )
    no_reduction: bool = (
        tokens_before is not None
        and tokens_after is not None
        and tokens_after >= tokens_before
    )
    if no_reduction or compression_count <= 0:
        headline = "Operational Checkpoint made no further reduction"
    else:
        headline = "Operational Checkpoint reduced active context budget"
    return {
        "headline": headline,
        "noop": no_reduction,
        "note": "Message counts are internal here; the useful number is active context budget.",
        "token_line": (
            f"Active context budget: {_fmt_tokens(tokens_before)} → "
            f"{_fmt_tokens(tokens_after)} tokens"
        ),
    }


def _install_tui_session_compress_response_hook(
    server_module: ModuleType | None = None,
) -> bool:
    module: ModuleType | None = (
        server_module if server_module is not None else sys.modules.get("tui_gateway.server")
    )
    if module is None:
        return False
    methods: object = getattr(module, "_methods", None)
    sessions: object = getattr(module, "_sessions", None)
    original_status_update: object = getattr(module, "_status_update", None)
    if not isinstance(methods, dict) or not isinstance(sessions, dict):
        return False
    if not callable(original_status_update):
        return False
    original_handler: object = methods.get("session.compress")
    if not callable(original_handler):
        return False
    if getattr(original_handler, "_operational_checkpoint_tui_compress_hook", False):
        return True

    def patched_session_compress(rid: object, params: dict[str, object]) -> dict[str, object]:
        sid: str = string_or_empty(params.get("session_id")).strip()
        focus_topic: str = string_or_empty(params.get("focus_topic")).strip()
        session: object = sessions.get(sid) if sid else None
        agent: object | None = session.get("agent") if isinstance(session, dict) else None
        start_tokens: int | None = None
        if agent is not None:
            engine: object = getattr(agent, "context_compressor", None)
            history: object = session.get("history") if isinstance(session, dict) else None
            if is_compaction_callback_engine(engine) and isinstance(history, list):
                start_tokens = _current_request_tokens(
                    agent=agent,
                    engine=engine,
                    messages=history,
                    system_prompt=_active_system_prompt(agent, ""),
                )

        status_was_wrapped: bool = False
        if start_tokens is not None:

            def plugin_status_update(
                status_sid: str,
                kind: str,
                text: str | None = None,
            ) -> object:
                if status_sid == sid and kind == "compressing":
                    line = (
                        "🗜️  Operational Checkpoint: compacting active context budget "
                        f"{_fmt_tokens(start_tokens)} tokens"
                    )
                    if focus_topic:
                        line = f'{line}, focus: "{focus_topic}"'
                    return original_status_update(status_sid, kind, line + "...")
                return original_status_update(status_sid, kind, text)

            setattr(module, "_status_update", plugin_status_update)
            status_was_wrapped = True

        try:
            with compaction_trigger_scope("manual"), swallow_compaction_preview():
                response: object = original_handler(rid, params)
        finally:
            if status_was_wrapped:
                setattr(module, "_status_update", original_status_update)

        if not isinstance(response, dict) or agent is None:
            return response if isinstance(response, dict) else {}
        result: object = response.get("result")
        if not isinstance(result, dict):
            return response
        artifact: dict[str, object] | None = latest_compaction_artifact_for_agent(agent)
        if artifact is None:
            return response
        tokens_before_raw: object = artifact.get("tokens_before")
        tokens_after_raw: object = artifact.get("tokens_after")
        if isinstance(tokens_before_raw, int):
            result["before_tokens"] = tokens_before_raw
        if isinstance(tokens_after_raw, int):
            result["after_tokens"] = tokens_after_raw
        result["summary"] = _tui_compaction_summary(artifact)
        result["operational_checkpoint"] = artifact
        return response

    setattr(patched_session_compress, "_operational_checkpoint_tui_compress_hook", True)
    methods["session.compress"] = patched_session_compress
    return True


def _normalize_summary_preview(summary: str) -> str:
    compact: str = " ".join(summary.split())
    if not compact:
        return ""
    preview_chars: int = _summary_preview_chars()
    if len(compact) <= preview_chars:
        return compact
    return f"{compact[: max(1, preview_chars - 1)].rstrip()}…"


def _record_compaction_artifact(
    *,
    agent: object,
    payload: dict[str, str | None],
) -> None:
    focus_topic: str | None = string_or_empty(payload.get("focus_topic")).strip() or None
    summary: str = string_or_empty(payload.get("summary")).strip()
    state: dict[str, object] = current_compaction_state() or {}
    artifact: dict[str, object] = {
        "focus_topic": focus_topic,
        "summary": summary,
        "trigger": current_compaction_trigger(),
        "timestamp": time.time(),
        "session_id": string_or_empty(getattr(agent, "session_id", "")) or None,
        "message_count_before": state.get("message_count_before"),
        "message_count_after": state.get("message_count_after"),
        "tokens_before": state.get("tokens_before"),
        "tokens_after": state.get("tokens_after"),
        "compression_count": state.get("compression_count"),
    }
    agent_state: AgentRuntimeState = _agent_runtime_state(agent)
    agent_state.last_compaction_artifact = artifact
    cli: object | None = agent_state.cli
    if cli is not None:
        _cli_runtime_state(cli).last_compaction_artifact = artifact

    if should_suppress_preview() or not _should_show_summary_preview() or not summary:
        return

    preview_line: str = f"     checkpoint: {_normalize_summary_preview(summary)}"
    if state:
        state["summary_preview_line"] = preview_line
        return

    _safe_emit(agent, preview_line)


def _bind_compaction_callback(cli: object, agent: object) -> None:
    engine: object = getattr(agent, "context_compressor", None)
    if not is_compaction_callback_engine(engine):
        return

    def callback(payload: dict[str, str | None]) -> None:
        _record_compaction_artifact(agent=agent, payload=payload)

    engine.set_compaction_callback(callback)
    _bind_agent_runtime_hooks(agent)
    _bind_engine_session_state(agent, engine)
    _agent_runtime_state(agent).cli = cli
    _activate_hydrated_session_state(cli, agent, engine)


def _sync_cli_session_id(agent: object) -> None:
    cli: object | None = _agent_runtime_state(agent).cli
    if not is_cli_bridge_target(cli):
        return
    session_id: str = string_or_empty(getattr(agent, "session_id", "")).strip()
    if not session_id:
        return
    cli.session_id = session_id


def _emit_start_status(
    *,
    agent: object,
    trigger: Literal["auto", "manual"],
    approx_tokens: int | None,
    focus_topic: str | None,
    threshold_tokens: int | None,
) -> None:
    if not _should_emit_status() or trigger == "manual":
        return

    threshold_text: str = ""
    if isinstance(threshold_tokens, int) and threshold_tokens > 0:
        threshold_text = f" / {threshold_tokens:,}"
    line: str = (
        "🗜️  Operational Checkpoint: auto-compacting "
        f"{_fmt_tokens(approx_tokens)}{threshold_text} tokens"
    )
    if focus_topic:
        line = f'{line}, focus: "{focus_topic}"'
    _safe_emit(agent, line + "...")


@dataclass(slots=True)
class CompactionStatus:
    compression_count_after: int
    compression_count_before: int
    message_count_after: int
    message_count_before: int
    summary_preview_line: str | None
    tokens_after: int | None
    tokens_before: int | None


def _emit_end_status(
    *,
    agent: object,
    trigger: Literal["auto", "manual"],
    status: CompactionStatus,
) -> None:
    if not _should_emit_status() or trigger == "manual":
        return

    if status.compression_count_after <= status.compression_count_before:
        _safe_emit(agent, "  🗜️  Operational Checkpoint made no further reduction.")
        return

    _safe_emit(
        agent,
        (
            "  ✅ Operational Checkpoint reduced active context budget: "
            f"{_fmt_tokens(status.tokens_before)} → {_fmt_tokens(status.tokens_after)} tokens"
        ),
    )
    if status.summary_preview_line:
        _safe_emit(agent, status.summary_preview_line)


def _emit_failure_status(
    agent: object,
    trigger: Literal["auto", "manual"],
    exc: Exception,
) -> None:
    if not _should_emit_status() or trigger == "manual":
        return
    _safe_emit(agent, f"  ❌ Operational Checkpoint failed: {exc}")


def _load_original_init_agent(
    cli_class: type[CLIBridgeTarget],
) -> Callable[..., bool]:
    candidate: object = cli_class.__dict__.get("_init_agent")
    if not callable(candidate):
        raise TypeError("Hermes CLI _init_agent is unavailable")

    def original_init_agent(
        self: CLIBridgeTarget,
        *args: object,
        **kwargs: object,
    ) -> bool:
        return bool(candidate(self, *args, **kwargs))

    return original_init_agent


def _load_original_manual_compress(
    cli_class: type[CLIBridgeTarget],
) -> Callable[..., None]:
    candidate: object = cli_class.__dict__.get("_manual_compress")
    if not callable(candidate):
        raise TypeError("Hermes CLI _manual_compress is unavailable")

    def original_manual_compress(
        self: CLIBridgeTarget,
        cmd_original: str = "",
    ) -> None:
        candidate(self, cmd_original)

    return original_manual_compress


def _load_original_preload_resumed_session(
    cli_class: type[CLIBridgeTarget],
) -> Callable[..., bool] | None:
    candidate: object = cli_class.__dict__.get("_preload_resumed_session")
    if not callable(candidate):
        return None

    def original_preload_resumed_session(self: CLIBridgeTarget) -> bool:
        return bool(candidate(self))

    return original_preload_resumed_session


def _load_original_handle_resume_command(
    cli_class: type[CLIBridgeTarget],
) -> Callable[..., None] | None:
    candidate: object = cli_class.__dict__.get("_handle_resume_command")
    if not callable(candidate):
        return None

    def original_handle_resume_command(
        self: CLIBridgeTarget,
        cmd_original: str,
    ) -> None:
        candidate(self, cmd_original)

    return original_handle_resume_command


def _load_original_agent_init(
    agent_class: type[AgentBridgeTarget],
) -> Callable[..., None] | None:
    candidate: object = agent_class.__dict__.get("__init__")
    if not callable(candidate):
        return None

    def original_agent_init(
        self: AgentBridgeTarget,
        *args: object,
        **kwargs: object,
    ) -> None:
        candidate(self, *args, **kwargs)

    return original_agent_init


def _load_original_compress_context(
    agent_class: type[AgentBridgeTarget],
) -> Callable[..., tuple[object, object]]:
    candidate: object = agent_class.__dict__.get("_compress_context")
    if not callable(candidate):
        raise TypeError("Hermes agent _compress_context is unavailable")

    def original_compress_context(
        self: AgentBridgeTarget,
        messages: list[dict[str, object]],
        system_message: str,
        **runtime_kwargs: object,
    ) -> tuple[object, object]:
        result: object = candidate(
            self,
            messages,
            system_message,
            **runtime_kwargs,
        )
        if (
            not isinstance(result, tuple)
            or len(result) != _COMPRESS_CONTEXT_RESULT_LEN
        ):
            raise TypeError("Hermes agent _compress_context returned an invalid payload")
        return result

    return original_compress_context


def _load_original_flush_messages_to_session_db(
    agent_class: type[AgentBridgeTarget],
) -> Callable[..., None] | None:
    candidate: object = agent_class.__dict__.get("_flush_messages_to_session_db")
    if not callable(candidate):
        return None

    def original_flush_messages_to_session_db(
        self: AgentBridgeTarget,
        messages: list[dict[str, object]],
        conversation_history: list[dict[str, object]] | None = None,
    ) -> None:
        candidate(self, messages, conversation_history)

    return original_flush_messages_to_session_db


def _load_original_save_session_log(
    agent_class: type[AgentBridgeTarget],
) -> Callable[..., None] | None:
    candidate: object = agent_class.__dict__.get("_save_session_log")
    if not callable(candidate):
        return None

    def original_save_session_log(
        self: AgentBridgeTarget,
        messages: list[dict[str, object]] | None = None,
    ) -> None:
        candidate(self, messages)

    return original_save_session_log


def _build_patched_init_agent(
    original_init_agent: Callable[..., bool],
) -> Callable[..., bool]:
    def patched_init_agent(
        self: CLIBridgeTarget,
        *args: object,
        **kwargs: object,
    ) -> bool:
        result: bool = original_init_agent(self, *args, **kwargs)
        if not result:
            return result

        agent: object = getattr(self, "agent", None)
        if agent is None:
            return result

        _bind_compaction_callback(self, agent)
        return result

    return patched_init_agent


def _build_patched_agent_init(
    original_agent_init: Callable[..., None],
) -> Callable[..., None]:
    def patched_agent_init(
        self: AgentBridgeTarget,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_agent_init(self, *args, **kwargs)
        _bind_agent_runtime_hooks(self)

    return patched_agent_init


def _build_patched_manual_compress(
    original_manual_compress: Callable[..., None],
) -> Callable[..., None]:
    def patched_manual_compress(
        self: CLIBridgeTarget,
        cmd_original: str = "",
    ) -> None:
        with compaction_trigger_scope("manual"), swallow_compaction_preview():
            original_manual_compress(self, cmd_original)
        agent: object = getattr(self, "agent", None)
        if agent is not None:
            _sync_cli_session_id(agent)

    return patched_manual_compress


def _build_patched_preload_resumed_session(
    original_preload_resumed_session: Callable[..., bool],
) -> Callable[..., bool]:
    def patched_preload_resumed_session(self: CLIBridgeTarget) -> bool:
        result: bool = original_preload_resumed_session(self)
        if result:
            _hydrate_cli_history_from_plugin_state(
                self,
                hermes_home=getattr(self, "hermes_home", None),
            )
        return result

    return patched_preload_resumed_session


def _build_patched_handle_resume_command(
    original_handle_resume_command: Callable[..., None],
) -> Callable[..., None]:
    def patched_handle_resume_command(
        self: CLIBridgeTarget,
        cmd_original: str,
    ) -> None:
        original_handle_resume_command(self, cmd_original)
        agent: object = getattr(self, "agent", None)
        if agent is None:
            return
        engine: object = getattr(agent, "context_compressor", None)
        if not is_compaction_callback_engine(engine):
            return
        _activate_hydrated_session_state(self, agent, engine)

    return patched_handle_resume_command


def _build_patched_compress_context(
    original_compress_context: Callable[..., tuple[object, object]],
) -> Callable[..., tuple[object, object]]:
    def patched_compress_context(
        self: AgentBridgeTarget,
        messages: list[dict[str, object]],
        system_message: str,
        **runtime_kwargs: object,
    ) -> tuple[object, object]:

        approx_tokens_raw: object = runtime_kwargs.get("approx_tokens")
        approx_tokens: int | None = (
            approx_tokens_raw if isinstance(approx_tokens_raw, int) else None
        )
        focus_topic_raw: object = runtime_kwargs.get("focus_topic")
        focus_topic_text: str = string_or_empty(focus_topic_raw).strip()
        focus_topic: str | None = focus_topic_text or None
        task_id_text: str = string_or_empty(runtime_kwargs.get("task_id")).strip()
        task_id: str = task_id_text or "default"

        engine: object = getattr(self, "context_compressor", None)
        if not is_compaction_callback_engine(engine):
            return original_compress_context(
                self,
                messages,
                system_message,
                approx_tokens=approx_tokens,
                task_id=task_id,
                focus_topic=focus_topic,
            )

        trigger: Literal["auto", "manual"] = current_compaction_trigger()
        message_count_before: int = len(messages)
        raw_message_count_before: int = _raw_message_count_for_hydrated_messages(
            self,
            messages,
        )
        compression_count_before: int = int(getattr(engine, "compression_count", 0) or 0)
        threshold_tokens: int | None = getattr(engine, "threshold_tokens", None)
        request_tokens_before: int = _current_request_tokens(
            agent=self,
            engine=engine,
            messages=messages,
            system_prompt=_active_system_prompt(self, system_message),
        )
        _emit_start_status(
            agent=self,
            trigger=trigger,
            approx_tokens=request_tokens_before,
            focus_topic=focus_topic,
            threshold_tokens=threshold_tokens if isinstance(threshold_tokens, int) else None,
        )

        state: dict[str, object] = {
            "compression_count": compression_count_before,
            "focus_topic": focus_topic,
            "message_count_before": raw_message_count_before,
            "tokens_before": request_tokens_before,
            "trigger": trigger,
        }

        try:
            with compaction_state_scope(state):
                compress_result: tuple[object, object] = _perform_plugin_owned_compaction(
                    self,
                    messages=messages,
                    system_message=system_message,
                    approx_tokens=approx_tokens,
                    task_id=task_id,
                    focus_topic=focus_topic,
                )
                compressed: object
                new_system_prompt: object
                compressed, new_system_prompt = compress_result
        except Exception as exc:
            _emit_failure_status(self, trigger, exc)
            raise

        tokens_after_value: object = getattr(engine, "last_prompt_tokens", None)
        tokens_after: int | None = (
            tokens_after_value if isinstance(tokens_after_value, int) else None
        )
        compression_count_after: int = int(getattr(engine, "compression_count", 0) or 0)
        message_count_after: int = (
            len(compressed) if isinstance(compressed, list) else message_count_before
        )
        summary_preview_value: object = state.get("summary_preview_line")
        summary_preview_line: str | None = (
            summary_preview_value
            if isinstance(summary_preview_value, str) and summary_preview_value
            else None
        )
        state.update(
            {
                "compression_count": compression_count_after,
                "message_count_after": message_count_after,
                "tokens_after": tokens_after,
            }
        )
        if compression_count_after > compression_count_before:
            _sync_cli_session_id(self)
        agent_runtime_state: AgentRuntimeState = _agent_runtime_state(self)
        artifact_value: dict[str, object] | None = agent_runtime_state.last_compaction_artifact
        if artifact_value is not None:
            artifact_value.update(
                {
                    "compression_count": compression_count_after,
                    "message_count_after": message_count_after,
                    "tokens_after": tokens_after,
                }
            )
        cli_value: object | None = agent_runtime_state.cli
        if cli_value is not None:
            cli_artifact_value: dict[str, object] | None = _cli_runtime_state(
                cli_value
            ).last_compaction_artifact
            if cli_artifact_value is not None:
                cli_artifact_value.update(
                    {
                        "compression_count": compression_count_after,
                        "message_count_after": message_count_after,
                        "tokens_after": tokens_after,
                    }
                )
        status = CompactionStatus(
            compression_count_after=compression_count_after,
            compression_count_before=compression_count_before,
            message_count_after=message_count_after,
            message_count_before=raw_message_count_before,
            summary_preview_line=summary_preview_line,
            tokens_after=tokens_after,
            tokens_before=request_tokens_before,
        )
        _emit_end_status(
            agent=self,
            trigger=trigger,
            status=status,
        )
        return compressed, new_system_prompt

    return patched_compress_context


def _build_patched_flush_messages_to_session_db(
    original_flush_messages_to_session_db: Callable[..., None],
) -> Callable[..., None]:
    def patched_flush_messages_to_session_db(
        self: AgentBridgeTarget,
        messages: list[dict[str, object]],
        conversation_history: list[dict[str, object]] | None = None,
    ) -> None:
        record: PersistedCompactionState | None = _active_compaction_state(self)
        if record is None:
            original_flush_messages_to_session_db(self, messages, conversation_history)
            return

        session_db: object = getattr(self, "_session_db", None)
        session_id: str = string_or_empty(getattr(self, "session_id", "")).strip()
        if session_db is None or not session_id:
            return

        apply_override: object = getattr(self, "_apply_persist_user_message_override", None)
        if callable(apply_override):
            apply_override(messages)

        try:
            ensure_session: object = getattr(session_db, "ensure_session", None)
            if callable(ensure_session):
                ensure_session(
                    session_id,
                    source=string_or_empty(getattr(self, "platform", "")) or "cli",
                    model=string_or_empty(getattr(self, "model", "")),
                )

            compacted_base_count: int = len(record.compacted_messages)
            runtime_state: AgentRuntimeState = _agent_runtime_state(self)
            flush_cursor_raw: int | None = runtime_state.hydrated_cursor
            flush_cursor: int = (
                flush_cursor_raw
                if isinstance(flush_cursor_raw, int)
                and flush_cursor_raw >= compacted_base_count
                else compacted_base_count
            )
            flush_from: int = max(compacted_base_count, flush_cursor)
            message: dict[str, object]
            for message in messages[flush_from:]:
                role: str = string_or_empty(message.get("role")).strip() or "unknown"
                tool_calls_data: object = None
                raw_tool_calls: object = message.get("tool_calls")
                if isinstance(raw_tool_calls, list):
                    tool_calls_data = raw_tool_calls
                append_message: object = getattr(session_db, "append_message", None)
                if not callable(append_message):
                    break
                append_message(
                    session_id=session_id,
                    role=role,
                    content=message.get("content"),
                    tool_name=message.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=message.get("tool_call_id"),
                    finish_reason=message.get("finish_reason"),
                    reasoning=message.get("reasoning") if role == "assistant" else None,
                    reasoning_details=(
                        message.get("reasoning_details") if role == "assistant" else None
                    ),
                    codex_reasoning_items=(
                        message.get("codex_reasoning_items") if role == "assistant" else None
                    ),
                )

            runtime_state.hydrated_cursor = len(messages)
            object.__setattr__(
                self,
                "_last_flushed_db_idx",
                record.raw_message_count + max(0, len(messages) - compacted_base_count),
            )
        except Exception as exc:
            logger.warning("Session DB append_message failed: %s", exc)

    return patched_flush_messages_to_session_db


def _build_patched_save_session_log(
    original_save_session_log: Callable[..., None],
) -> Callable[..., None]:
    def patched_save_session_log(
        self: AgentBridgeTarget,
        messages: list[dict[str, object]] | None = None,
    ) -> None:
        record: PersistedCompactionState | None = _active_compaction_state(self)
        if record is None or messages is None:
            original_save_session_log(self, messages)
            return
        raw_messages: list[dict[str, object]] = _raw_messages_for_session_log(
            agent=self,
            messages=messages,
            record=record,
        )
        original_save_session_log(self, raw_messages)

    return patched_save_session_log


def install_agent_runtime_bridge(
    *,
    agent_class: type[AgentBridgeTarget],
) -> None:
    bridge_installed: object = agent_class.__dict__.get(
        "_operational_checkpoint_agent_bridge_installed",
        False,
    )
    if bridge_installed:
        return

    original_agent_init: Callable[..., None] | None = _load_original_agent_init(agent_class)
    original_compress_context: Callable[..., tuple[object, object]] = (
        _load_original_compress_context(agent_class)
    )
    original_flush_messages_to_session_db: Callable[..., None] | None = (
        _load_original_flush_messages_to_session_db(agent_class)
    )
    original_save_session_log: Callable[..., None] | None = (
        _load_original_save_session_log(agent_class)
    )

    if original_agent_init is not None:
        type.__setattr__(
            agent_class,
            "__init__",
            _build_patched_agent_init(original_agent_init),
        )
    type.__setattr__(
        agent_class,
        "_compress_context",
        _build_patched_compress_context(original_compress_context),
    )
    if original_flush_messages_to_session_db is not None:
        type.__setattr__(
            agent_class,
            "_flush_messages_to_session_db",
            _build_patched_flush_messages_to_session_db(
                original_flush_messages_to_session_db
            ),
        )
    if original_save_session_log is not None:
        type.__setattr__(
            agent_class,
            "_save_session_log",
            _build_patched_save_session_log(original_save_session_log),
        )
    type.__setattr__(
        agent_class,
        "_operational_checkpoint_agent_bridge_installed",
        True,
    )


def install_runtime_bridge(
    *,
    cli_class: type[CLIBridgeTarget],
    agent_class: type[AgentBridgeTarget],
) -> None:
    install_agent_runtime_bridge(agent_class=agent_class)

    bridge_installed: object = cli_class.__dict__.get(
        "_operational_checkpoint_bridge_installed",
        False,
    )
    if bridge_installed:
        return

    original_init_agent: Callable[..., bool] = _load_original_init_agent(cli_class)
    original_manual_compress: Callable[..., None] = _load_original_manual_compress(
        cli_class
    )
    original_preload_resumed_session: Callable[..., bool] | None = (
        _load_original_preload_resumed_session(cli_class)
    )
    original_handle_resume_command: Callable[..., None] | None = (
        _load_original_handle_resume_command(cli_class)
    )

    type.__setattr__(
        cli_class,
        "_init_agent",
        _build_patched_init_agent(original_init_agent),
    )
    type.__setattr__(
        cli_class,
        "_manual_compress",
        _build_patched_manual_compress(original_manual_compress),
    )
    if original_preload_resumed_session is not None:
        type.__setattr__(
            cli_class,
            "_preload_resumed_session",
            _build_patched_preload_resumed_session(
                original_preload_resumed_session
            ),
        )
    if original_handle_resume_command is not None:
        type.__setattr__(
            cli_class,
            "_handle_resume_command",
            _build_patched_handle_resume_command(original_handle_resume_command),
        )
    type.__setattr__(
        cli_class,
        "_operational_checkpoint_bridge_installed",
        True,
    )


def _resolve_agent_target(
    *,
    run_agent_module: ModuleType | None = None,
) -> type[AgentBridgeTarget] | None:
    resolved_run_agent_module: ModuleType | None = (
        run_agent_module
        if run_agent_module is not None
        else sys.modules.get("run_agent")
    )
    if resolved_run_agent_module is None:
        return None

    ai_agent_candidate: object = getattr(
        resolved_run_agent_module,
        "AIAgent",
        None,
    )
    if not isinstance(ai_agent_candidate, type):
        return None
    return ai_agent_candidate


def _resolve_bridge_targets(
    *,
    cli_module: ModuleType | None = None,
    run_agent_module: ModuleType | None = None,
) -> tuple[type[CLIBridgeTarget], type[AgentBridgeTarget]] | None:
    resolved_cli_module: ModuleType | None = (
        cli_module if cli_module is not None else sys.modules.get("cli")
    )
    if resolved_cli_module is None:
        return None

    ai_agent_candidate: type[AgentBridgeTarget] | None = _resolve_agent_target(
        run_agent_module=run_agent_module
    )
    hermes_cli_candidate: object = getattr(
        resolved_cli_module,
        "HermesCLI",
        None,
    )
    if not isinstance(hermes_cli_candidate, type) or ai_agent_candidate is None:
        return None

    return hermes_cli_candidate, ai_agent_candidate


def _attempt_sidecar_install() -> bool:
    _install_tui_session_compress_response_hook()
    ai_agent_class: type[AgentBridgeTarget] | None = _resolve_agent_target()
    if ai_agent_class is not None:
        install_agent_runtime_bridge(agent_class=ai_agent_class)

    bridge_targets: tuple[type[CLIBridgeTarget], type[AgentBridgeTarget]] | None = (
        _resolve_bridge_targets()
    )
    if bridge_targets is None:
        return False

    hermes_cli_class: type[CLIBridgeTarget]
    hermes_agent_class: type[AgentBridgeTarget]
    hermes_cli_class, hermes_agent_class = bridge_targets
    install_runtime_bridge(cli_class=hermes_cli_class, agent_class=hermes_agent_class)
    return True


def _deferred_install_worker() -> None:
    global _SIDECAR_INSTALL_SCHEDULED

    deadline: float = time.monotonic() + _DEFERRED_INSTALL_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if _attempt_sidecar_install():
                return
            time.sleep(_DEFERRED_INSTALL_POLL_INTERVAL_SECONDS)
        logger.debug(
            "Operational Checkpoint sidecar install timed out waiting for cli.HermesCLI"
        )
    except Exception:
        logger.exception("Operational Checkpoint deferred sidecar install failed")
    finally:
        with _SIDECAR_INSTALL_LOCK:
            _SIDECAR_INSTALL_SCHEDULED = False


def _schedule_deferred_install() -> None:
    global _SIDECAR_INSTALL_SCHEDULED

    with _SIDECAR_INSTALL_LOCK:
        if _SIDECAR_INSTALL_SCHEDULED:
            return
        _SIDECAR_INSTALL_SCHEDULED = True

    install_thread = threading.Thread(
        target=_deferred_install_worker,
        name="operational-checkpoint-sidecar-install",
        daemon=True,
    )
    install_thread.start()


def install_plugin_sidecar() -> None:
    if _attempt_sidecar_install():
        return

    # Hermes can now discover entry-point plugins while only run_agent is
    # imported. In that ordering, waiting for ``cli`` to already exist drops the
    # bridge permanently: the context engine registers, but the CLI/manual and
    # auto-compaction sidecar patches never install. Always schedule the short
    # deferred probe after an immediate miss so the bridge can attach when the
    # complementary module finishes importing.
    _schedule_deferred_install()
