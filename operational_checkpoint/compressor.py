from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from agent.context_engine import ContextEngine
from agent.model_metadata import estimate_messages_tokens_rough

from .helpers import (
    LEGACY_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
    as_positive_int,
    call_llm,
    estimate_tokens,
    extract_choice_content,
    load_runtime_defaults,
    load_usage_snapshots,
    require_non_negative_int,
    require_positive_int,
    resolve_hermes_home,
    save_usage_snapshots,
    string_or_empty,
    strip_summary_prefix,
)
from .prompt import (
    OPERATIONAL_CHECKPOINT_SUMMARIZER_PREAMBLE,
    OPERATIONAL_CHECKPOINT_TEMPLATE,
)
from .sidecar import install_plugin_sidecar

logger: logging.Logger = logging.getLogger(__name__)

_MINIMUM_MESSAGES_TO_COMPACT: int = 1
_CHECKPOINT_SEPARATOR: str = (
    "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
)
_REQUIRED_CHECKPOINT_SECTIONS: tuple[str, ...] = (
    "Objective",
    "Explicit user instructions / prohibitions / scope boundaries",
    "Operational state",
    "Active working set",
    "Discoveries / evidence",
    "Settled decisions / rejected alternatives",
    "Transferable patterns learned this run",
    "Assumptions / uncertainties / blockers",
    "Execution status",
    "Action frontier",
    "Critical invariants / regression risks",
)
_ALLOWED_EPISTEMIC_LABELS: frozenset[str] = frozenset(
    {"Observed", "Inferred", "Assumption", "Unknown", "Blocked"}
)
_BRACKETED_LABEL_RE: re.Pattern[str] = re.compile(r"\[([^\]]+)\]")


class ToolCallFunction(TypedDict, total=False):
    name: str
    arguments: str


class ToolCallPayload(TypedDict, total=False):
    id: str
    function: ToolCallFunction


class ChatMessage(TypedDict, total=False):
    role: str
    content: str
    tool_call_id: str
    tool_calls: list[ToolCallPayload]


@dataclass(frozen=True)
class SummaryRequest:
    api_key: str | None
    base_url: str | None
    extra_body: dict[str, object] | None
    input_tokens_estimate: int
    main_runtime: dict[str, object] | None
    model: str | None
    previous_summary_tokens: int
    provider: str | None
    prompt: str


@dataclass(frozen=True)
class CompactionWindow:
    head: list[ChatMessage]
    middle: list[ChatMessage]
    tail: list[ChatMessage]
    compressed_count: int
def _coerce_message(message: Mapping[str, object]) -> ChatMessage:
    coerced: ChatMessage = {}
    role: str = string_or_empty(message.get("role")).strip()
    if role:
        coerced["role"] = role
    if "content" in message:
        coerced["content"] = string_or_empty(message.get("content"))
    tool_call_id: str = string_or_empty(message.get("tool_call_id")).strip()
    if tool_call_id:
        coerced["tool_call_id"] = tool_call_id

    raw_tool_calls: object = message.get("tool_calls")
    tool_calls: list[ToolCallPayload] = []
    if isinstance(raw_tool_calls, list):
        raw_tool_call: object
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            function_payload: ToolCallFunction = {}
            raw_function: object = raw_tool_call.get("function")
            if isinstance(raw_function, dict):
                function_name: str = string_or_empty(raw_function.get("name")).strip()
                if function_name:
                    function_payload["name"] = function_name
                function_arguments: str = string_or_empty(
                    raw_function.get("arguments")
                )
                if function_arguments:
                    function_payload["arguments"] = function_arguments
            tool_call: ToolCallPayload = {}
            tool_call_id_value: str = string_or_empty(raw_tool_call.get("id")).strip()
            if tool_call_id_value:
                tool_call["id"] = tool_call_id_value
            if function_payload:
                tool_call["function"] = function_payload
            if tool_call:
                tool_calls.append(tool_call)
    if tool_calls:
        coerced["tool_calls"] = tool_calls
    return coerced


def _copy_messages(messages: Sequence[Mapping[str, object]]) -> list[ChatMessage]:
    return [_coerce_message(message) for message in messages if isinstance(message, dict)]


def _message_role(message: Mapping[str, object]) -> str:
    return string_or_empty(message.get("role")).strip().lower()


def _message_content(message: Mapping[str, object]) -> str:
    return string_or_empty(message.get("content"))


def _tool_calls(message: Mapping[str, object]) -> list[ToolCallPayload]:
    raw_tool_calls: object = message.get("tool_calls")
    tool_calls: list[ToolCallPayload] = []
    if not isinstance(raw_tool_calls, list):
        return tool_calls
    raw_tool_call: object
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        function_payload: ToolCallFunction = {}
        raw_function: object = raw_tool_call.get("function")
        if isinstance(raw_function, dict):
            function_name: str = string_or_empty(raw_function.get("name")).strip()
            if function_name:
                function_payload["name"] = function_name
            function_arguments: str = string_or_empty(raw_function.get("arguments"))
            if function_arguments:
                function_payload["arguments"] = function_arguments
        tool_call: ToolCallPayload = {}
        tool_call_id_value: str = string_or_empty(raw_tool_call.get("id")).strip()
        if tool_call_id_value:
            tool_call["id"] = tool_call_id_value
        if function_payload:
            tool_call["function"] = function_payload
        if tool_call:
            tool_calls.append(tool_call)
    return tool_calls


def _tool_call_id(tool_call: ToolCallPayload) -> str:
    return string_or_empty(tool_call.get("id")).strip()


def _runtime_messages(messages: list[ChatMessage]) -> list[dict[str, object]]:
    return [dict(message) for message in messages]


class OperationalCheckpointCompressor(ContextEngine):
    api_key: str
    api_mode: str
    auto_compact_at_tokens: int
    base_url: str
    bound_session_id: str
    compression_count: int
    context_length: int
    head_preserve_messages: int
    hermes_home: Path
    last_checkpoint_focus_topic: str | None
    last_checkpoint_summary: str
    last_completion_tokens: int
    last_prompt_tokens: int
    last_total_tokens: int
    minimum_tail_messages: int
    model: str
    protect_first_n: int
    protect_last_n: int
    provider: str
    reasoning_effort: str
    runtime_api_key: str
    runtime_api_mode: str
    runtime_base_url: str
    runtime_model: str
    runtime_provider: str
    summary_retry_attempts: int
    tail_preserve_tokens: int
    tail_token_budget: int
    threshold_percent: float
    threshold_tokens: int
    _compaction_callback: Callable[[dict[str, str | None]], None] | None
    _previous_summary: str

    @property
    def name(self) -> str:
        return "operational_checkpoint"

    def __init__(self) -> None:
        self._compaction_callback = None
        self.bound_session_id = ""
        self.hermes_home = resolve_hermes_home()
        self.last_checkpoint_focus_topic = None
        self.last_checkpoint_summary = ""
        self.last_completion_tokens = 0
        self.last_prompt_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self._previous_summary = ""
        self.api_key = ""
        self.api_mode = ""
        self.base_url = ""
        self.model = ""
        self.provider = ""
        self.runtime_api_key = ""
        self.runtime_api_mode = ""
        self.runtime_base_url = ""
        self.runtime_model = ""
        self.runtime_provider = ""
        self.reasoning_effort = ""
        self.auto_compact_at_tokens = 0
        self.context_length = 0
        self.head_preserve_messages = 0
        self.minimum_tail_messages = 0
        self.tail_preserve_tokens = 0
        self.tail_token_budget = 0
        self.threshold_percent = 0.0
        self.threshold_tokens = 0
        self.protect_first_n = 0
        self.protect_last_n = 0
        self.summary_retry_attempts = 0
        self._reload_from_config(model_override=None, provider_override=None)

    def _reload_from_config(
        self,
        *,
        model_override: str | None,
        provider_override: str | None,
    ) -> None:
        defaults: dict[str, object] = load_runtime_defaults()
        context_limit_tokens: int = require_positive_int(
            defaults.get("context_limit_tokens"),
            "context_limit_tokens",
        )
        raw_threshold_percent: object = defaults.get("compaction_threshold_percent")
        if raw_threshold_percent is None:
            legacy_threshold_tokens = require_positive_int(
                defaults.get("auto_compact_at_tokens"),
                "auto_compact_at_tokens",
            )
            compaction_threshold_percent = legacy_threshold_tokens / context_limit_tokens
        else:
            compaction_threshold_percent = float(raw_threshold_percent)
        auto_compact_at_tokens: int = max(
            1,
            int(context_limit_tokens * compaction_threshold_percent),
        )
        head_preserve_messages: int = require_non_negative_int(
            defaults.get("head_preserve_messages"),
            "head_preserve_messages",
        )
        minimum_tail_messages: int = require_non_negative_int(
            defaults.get("minimum_tail_messages"),
            "minimum_tail_messages",
        )
        tail_preserve_tokens: int = require_non_negative_int(
            defaults.get("tail_preserve_tokens"),
            "tail_preserve_tokens",
        )

        self.model = (
            model_override.strip()
            if isinstance(model_override, str) and model_override.strip()
            else string_or_empty(defaults.get("model")).strip()
        )
        if provider_override is not None:
            self.provider = string_or_empty(provider_override).strip()
        else:
            self.provider = string_or_empty(defaults.get("provider")).strip()
        self.base_url = string_or_empty(defaults.get("base_url"))
        self.auto_compact_at_tokens = auto_compact_at_tokens
        self.context_length = context_limit_tokens
        self.head_preserve_messages = head_preserve_messages
        self.minimum_tail_messages = minimum_tail_messages
        self.protect_first_n = head_preserve_messages
        self.protect_last_n = minimum_tail_messages
        self.reasoning_effort = string_or_empty(defaults.get("reasoning_effort")).strip()
        self.summary_retry_attempts = require_positive_int(
            defaults.get("summary_retry_attempts"),
            "summary_retry_attempts",
        )
        self.tail_preserve_tokens = tail_preserve_tokens
        self.tail_token_budget = tail_preserve_tokens
        self.threshold_percent = compaction_threshold_percent
        self.threshold_tokens = auto_compact_at_tokens

    @staticmethod
    def _coerce_non_negative_int(value: object, default: int = 0) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value if value >= 0 else default
        if isinstance(value, str) and value.strip():
            try:
                parsed: int = int(value)
            except ValueError:
                return default
            return parsed if parsed >= 0 else default
        return default

    @staticmethod
    def _with_summary_prefix(summary: str) -> str:
        text: str = (summary or "").strip()
        prefix: str
        for prefix in (LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX):
            if text.startswith(prefix):
                text = text[len(prefix) :].lstrip()
                break
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    @staticmethod
    def _canonical_section_line(index: int, title: str) -> str:
        return f"{index}. {title}"

    @classmethod
    def _summary_validation_errors(cls, summary: str) -> list[str]:
        text: str = (summary or "").strip()
        if not text:
            return ["summary is empty"]

        errors: list[str] = []
        search_start: int = 0
        section_index: int
        section_title: str
        for section_index, section_title in enumerate(_REQUIRED_CHECKPOINT_SECTIONS, start=1):
            canonical_heading: str = cls._canonical_section_line(section_index, section_title)
            heading_position: int = text.find(canonical_heading, search_start)
            if heading_position < 0:
                errors.append(f"missing section: {canonical_heading}")
                continue
            search_start = heading_position + len(canonical_heading)

        raw_label: str
        for raw_label in _BRACKETED_LABEL_RE.findall(text):
            normalized_label: str = raw_label.strip()
            if normalized_label and normalized_label not in _ALLOWED_EPISTEMIC_LABELS:
                errors.append(f"invalid epistemic label: [{normalized_label}]")

        return errors

    @classmethod
    def _fallback_section_body(cls, section_index: int) -> str:
        if section_index == 1:
            return "- [Observed] Continue from the recent tail after compaction."
        if section_index == 3:
            return "- [Observed] Summary generation was unavailable."
        if section_index == 5:
            return "- [Observed] Earlier turns were compacted without a full LLM-generated checkpoint."
        if section_index == 8:
            return (
                "- [Unknown] The compacted middle-turn state was not summarized by "
                "the LLM; re-verify critical assumptions from repo/runtime state."
            )
        if section_index == 9:
            return (
                "- Accomplished: [Observed] Compaction completed with plugin-owned "
                "fallback checkpoint.\n"
                "- In progress: [Unknown] Exact state from compacted middle turns.\n"
                "- Remaining: [Observed] Continue from recent messages and inspect "
                "live artifacts before acting on old assumptions."
            )
        if section_index == 10:
            return (
                "- [Observed] Continue from the recent messages below and verify "
                "state from repo/runtime before redoing work."
            )
        if section_index == 11:
            return (
                "- [Observed] Treat this fallback checkpoint as incomplete; do not "
                "rely on it for decisions that need exact compacted-turn evidence."
            )
        return "None."

    @classmethod
    def _fallback_checkpoint_body(cls, dropped_turn_count: int) -> str:
        sections: list[str] = []
        section_index: int
        section_title: str
        for section_index, section_title in enumerate(_REQUIRED_CHECKPOINT_SECTIONS, start=1):
            body: str = cls._fallback_section_body(section_index)
            if section_index == 5:
                body += (
                    f"\n- [Observed] {dropped_turn_count} earlier turns were compacted "
                    "without a full structured checkpoint."
                )
            sections.append(
                f"{cls._canonical_section_line(section_index, section_title)}\n{body}"
            )
        return "\n\n".join(sections)

    def _snapshot_payload(self) -> dict[str, object]:
        total_tokens: int = self.last_total_tokens
        if total_tokens <= 0:
            total_tokens = self.last_prompt_tokens + self.last_completion_tokens
        return {
            "compression_count": int(self.compression_count or 0),
            "last_checkpoint_focus_topic": self.last_checkpoint_focus_topic,
            "last_checkpoint_summary": self.last_checkpoint_summary,
            "last_completion_tokens": int(self.last_completion_tokens or 0),
            "last_prompt_tokens": int(self.last_prompt_tokens or 0),
            "last_total_tokens": total_tokens,
            "updated_at": time.time(),
        }

    def bind_session(
        self,
        *,
        session_id: str,
        hermes_home: str | None = None,
        parent_session_id: str | None = None,
    ) -> None:
        del parent_session_id
        self.bound_session_id = string_or_empty(session_id).strip()
        self.hermes_home = resolve_hermes_home(hermes_home or self.hermes_home)

    def restore_usage_snapshot(self) -> bool:
        session_id: str = self.bound_session_id
        if not session_id:
            return False
        snapshot: dict[str, object] | None = load_usage_snapshots(self.hermes_home).get(
            session_id
        )
        if snapshot is None:
            return False

        self.last_prompt_tokens = self._coerce_non_negative_int(
            snapshot.get("last_prompt_tokens"),
            0,
        )
        self.last_completion_tokens = self._coerce_non_negative_int(
            snapshot.get("last_completion_tokens"),
            0,
        )
        self.last_total_tokens = self._coerce_non_negative_int(
            snapshot.get("last_total_tokens"),
            self.last_prompt_tokens + self.last_completion_tokens,
        )
        self.compression_count = self._coerce_non_negative_int(
            snapshot.get("compression_count"),
            int(self.compression_count or 0),
        )
        self.last_checkpoint_focus_topic = (
            string_or_empty(snapshot.get("last_checkpoint_focus_topic")).strip() or None
        )
        self.last_checkpoint_summary = string_or_empty(
            snapshot.get("last_checkpoint_summary")
        ).strip()
        if self.last_checkpoint_summary:
            self._previous_summary = self.last_checkpoint_summary
        return True

    def persist_usage_snapshot(self) -> None:
        session_id: str = self.bound_session_id
        if not session_id:
            return
        snapshots: dict[str, dict[str, object]] = load_usage_snapshots(self.hermes_home)
        snapshots[session_id] = self._snapshot_payload()
        save_usage_snapshots(snapshots, self.hermes_home)

    def rollover_usage_snapshot(
        self,
        *,
        new_session_id: str,
        post_compaction_tokens: int,
    ) -> None:
        previous_session_id: str = self.bound_session_id
        if previous_session_id:
            self.persist_usage_snapshot()
        normalized_post_compaction_tokens: int = self._coerce_non_negative_int(
            post_compaction_tokens,
            0,
        )
        self.bind_session(
            session_id=new_session_id,
            hermes_home=str(self.hermes_home),
        )
        self.last_prompt_tokens = normalized_post_compaction_tokens
        self.last_completion_tokens = 0
        self.last_total_tokens = normalized_post_compaction_tokens
        self.persist_usage_snapshot()

    def on_session_start(self, session_id: str, **kwargs: object) -> None:
        self.bind_session(
            session_id=session_id,
            hermes_home=string_or_empty(kwargs.get("hermes_home")) or None,
        )
        self.restore_usage_snapshot()

    def on_session_reset(self) -> None:
        ContextEngine.on_session_reset(self)
        self.bound_session_id = ""
        self.last_checkpoint_focus_topic = None
        self.last_checkpoint_summary = ""
        self._previous_summary = ""

    def update_model(self, model: str, *runtime_args: object, **runtime_config: object) -> None:
        context_length_raw: object = runtime_config.get("context_length")
        if context_length_raw is None and runtime_args:
            context_length_raw = runtime_args[0]
        if context_length_raw is None:
            context_length_raw = load_runtime_defaults()["context_limit_tokens"]
        provider_override: str = string_or_empty(runtime_config.get("provider")).strip()
        base_url: str = string_or_empty(runtime_config.get("base_url"))
        api_key: str = string_or_empty(runtime_config.get("api_key"))
        api_mode: str = string_or_empty(runtime_config.get("api_mode"))
        self._reload_from_config(
            model_override=None,
            provider_override=provider_override,
        )
        self.runtime_model = string_or_empty(model).strip()
        self.runtime_provider = provider_override
        self.runtime_base_url = base_url
        self.runtime_api_key = api_key
        self.runtime_api_mode = api_mode
        self.base_url = base_url
        self.api_key = api_key
        self.api_mode = api_mode
        self.context_length = as_positive_int(context_length_raw, self.context_length)
        self.threshold_tokens = max(1, int(self.context_length * self.threshold_percent))
        self.auto_compact_at_tokens = self.threshold_tokens

    def update_from_response(self, usage: dict[str, object]) -> None:
        self.last_prompt_tokens = self._coerce_non_negative_int(
            usage.get("prompt_tokens"),
            0,
        )
        self.last_completion_tokens = self._coerce_non_negative_int(
            usage.get("completion_tokens"),
            0,
        )
        self.last_total_tokens = self._coerce_non_negative_int(
            usage.get("total_tokens"),
            self.last_prompt_tokens + self.last_completion_tokens,
        )
        self.persist_usage_snapshot()

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        tracked_tokens: int = self._coerce_non_negative_int(
            self.last_total_tokens,
            self.last_prompt_tokens + self.last_completion_tokens,
        )
        current_tokens: int = self._coerce_non_negative_int(
            prompt_tokens,
            tracked_tokens,
        )
        return current_tokens >= self.threshold_tokens

    def set_compaction_callback(
        self,
        callback: Callable[[dict[str, str | None]], None] | None,
    ) -> None:
        self._compaction_callback = callback

    def _serialize_for_summary(self, turns: Sequence[Mapping[str, object]]) -> str:
        parts: list[str] = []
        message: Mapping[str, object]
        for message in turns:
            role: str = _message_role(message)
            content: str = _message_content(message)

            if role == "tool":
                tool_id: str = string_or_empty(message.get("tool_call_id")).strip()
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            if role == "assistant":
                rendered_tool_calls: list[str] = []
                tool_call: ToolCallPayload
                for tool_call in _tool_calls(message):
                    raw_function_payload: object = tool_call.get("function")
                    function_payload: Mapping[str, object] = (
                        raw_function_payload
                        if isinstance(raw_function_payload, Mapping)
                        else {}
                    )
                    name: str = (
                        string_or_empty(function_payload.get("name")).strip() or "?"
                    )
                    arguments: str = string_or_empty(function_payload.get("arguments"))
                    rendered_tool_calls.append(f"  {name}({arguments})")
                if rendered_tool_calls:
                    content += "\n[Tool calls:\n" + "\n".join(rendered_tool_calls) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            parts.append(f"[{role.upper() or 'UNKNOWN'}]: {content}")

        return "\n\n".join(parts)

    def _generate_summary(
        self,
        turns_to_summarize: Sequence[Mapping[str, object]],
        focus_topic: str | None = None,
    ) -> str | None:
        summary_request: SummaryRequest = self._build_summary_request(
            turns_to_summarize,
            focus_topic=focus_topic,
        )
        last_runtime_error: RuntimeError | None = None
        attempt_index: int
        for attempt_index in range(self.summary_retry_attempts):
            try:
                response: object = call_llm(
                    task="compression",
                    provider=summary_request.provider,
                    model=summary_request.model,
                    base_url=summary_request.base_url,
                    api_key=summary_request.api_key,
                    messages=[{"role": "user", "content": summary_request.prompt}],
                    max_tokens=None,
                    extra_body=summary_request.extra_body,
                    main_runtime=summary_request.main_runtime,
                )
            except RuntimeError as exc:
                last_runtime_error = exc
                logger.warning(
                    "operational checkpoint summary attempt %d/%d failed: %s",
                    attempt_index + 1,
                    self.summary_retry_attempts,
                    exc,
                )
                continue
            try:
                content: str = extract_choice_content(response)
                summary_body: str = content.strip()
                if not summary_body:
                    logger.warning(
                        "operational checkpoint summary attempt %d/%d returned empty content",
                        attempt_index + 1,
                        self.summary_retry_attempts,
                    )
                    continue
                validation_errors: list[str] = self._summary_validation_errors(summary_body)
                if validation_errors:
                    logger.warning(
                        "operational checkpoint summary attempt %d/%d failed schema validation: %s",
                        attempt_index + 1,
                        self.summary_retry_attempts,
                        "; ".join(validation_errors[:5]),
                    )
                    continue
                summary_tokens_estimate: int = estimate_tokens(summary_body)
                self.last_checkpoint_focus_topic = focus_topic
                self.last_checkpoint_summary = summary_body
                self._previous_summary = summary_body
                logger.info(
                    (
                        "operational checkpoint summary metrics: focus=%s "
                        "turns=%d input_tokens_est=%d previous_summary_tokens_est=%d "
                        "summary_tokens_est=%d summary_to_input_ratio=%.3f"
                    ),
                    focus_topic or "none",
                    len(turns_to_summarize),
                    summary_request.input_tokens_estimate,
                    summary_request.previous_summary_tokens,
                    summary_tokens_estimate,
                    (
                        summary_tokens_estimate / summary_request.input_tokens_estimate
                        if summary_request.input_tokens_estimate > 0
                        else 0.0
                    ),
                )
                return self._with_summary_prefix(summary_body)
            except (AttributeError, IndexError, LookupError, TypeError, ValueError) as exc:
                logger.warning("operational checkpoint summary failed: %s", exc)
                return None

        logger.warning(
            "operational checkpoint summary unavailable after %d attempts%s",
            self.summary_retry_attempts,
            f": {last_runtime_error}" if last_runtime_error is not None else "",
        )
        return None

    def _build_summary_request(
        self,
        turns_to_summarize: Sequence[Mapping[str, object]],
        *,
        focus_topic: str | None,
    ) -> SummaryRequest:
        content_to_summarize: str = self._serialize_for_summary(turns_to_summarize)
        input_tokens_estimate: int = estimate_tokens(content_to_summarize)
        previous_summary_tokens: int = estimate_tokens(self._previous_summary)
        prompt: str
        if self._previous_summary:
            prompt = (
                f"{OPERATIONAL_CHECKPOINT_SUMMARIZER_PREAMBLE}\n\n"
                "You are updating an existing operational checkpoint after another "
                "compaction pass.\n\n"
                f"PREVIOUS CHECKPOINT:\n{self._previous_summary}\n\n"
                f"NEW TURNS TO INCORPORATE:\n{content_to_summarize}\n\n"
                "Preserve still-valid prior state, integrate the new turns, "
                "and update the execution boundary.\n\n"
                f"{OPERATIONAL_CHECKPOINT_TEMPLATE}"
            )
        else:
            prompt = (
                f"{OPERATIONAL_CHECKPOINT_SUMMARIZER_PREAMBLE}\n\n"
                f"TURNS TO COMPACT:\n{content_to_summarize}\n\n"
                f"{OPERATIONAL_CHECKPOINT_TEMPLATE}"
            )

        if focus_topic:
            prompt += (
                f"\n\nFOCUS TOPIC: {focus_topic}\n"
                "Prioritize preserving all information tightly related to that topic. "
                "Compress unrelated material more aggressively."
            )

        extra_body: dict[str, object] | None = None
        if self.reasoning_effort:
            extra_body = {"reasoning": {"effort": self.reasoning_effort}}

        provider_name: str = self.provider or ("custom" if self.base_url else "")
        explicit_base_url: str | None = None
        explicit_api_key: str | None = None
        if provider_name == "custom":
            explicit_base_url = self.base_url or None
            explicit_api_key = self.api_key or None
        main_runtime: dict[str, object] | None = None
        runtime_model: str = self.runtime_model or self.model
        runtime_provider: str = self.runtime_provider or provider_name
        runtime_base_url: str = self.runtime_base_url or self.base_url
        runtime_api_key: str = self.runtime_api_key or self.api_key
        runtime_api_mode: str = self.runtime_api_mode or self.api_mode
        if any((runtime_model, runtime_provider, runtime_base_url, runtime_api_key, runtime_api_mode)):
            main_runtime = {
                "model": runtime_model,
                "provider": runtime_provider,
                "base_url": runtime_base_url,
                "api_key": runtime_api_key,
                "api_mode": runtime_api_mode,
            }
        return SummaryRequest(
            api_key=explicit_api_key,
            base_url=explicit_base_url,
            extra_body=extra_body,
            input_tokens_estimate=input_tokens_estimate,
            main_runtime=main_runtime,
            model=self.model or None,
            previous_summary_tokens=previous_summary_tokens,
            provider=provider_name or None,
            prompt=prompt,
        )

    def _message_token_estimate(self, message: Mapping[str, object]) -> int:
        token_estimate: int = estimate_tokens(_message_content(message))
        tool_call: ToolCallPayload
        for tool_call in _tool_calls(message):
            arguments: str = string_or_empty(tool_call.get("function", {}).get("arguments"))
            token_estimate += estimate_tokens(arguments)
        return max(1, token_estimate)

    def _align_boundary_forward(
        self,
        messages: Sequence[Mapping[str, object]],
        boundary: int,
    ) -> int:
        while boundary < len(messages) and _message_role(messages[boundary]) == "tool":
            boundary += 1
        return boundary

    def _align_boundary_backward(
        self,
        messages: Sequence[Mapping[str, object]],
        boundary: int,
    ) -> int:
        if boundary <= 0 or boundary >= len(messages):
            return boundary
        check_index: int = boundary - 1
        while check_index >= 0 and _message_role(messages[check_index]) == "tool":
            check_index -= 1
        if (
            check_index >= 0
            and _message_role(messages[check_index]) == "assistant"
            and _tool_calls(messages[check_index])
        ):
            return check_index
        return boundary

    def _find_tail_start(
        self,
        messages: Sequence[Mapping[str, object]],
        head_end: int,
    ) -> int:
        token_budget: int = max(0, self.tail_token_budget)
        minimum_tail: int = min(
            self.minimum_tail_messages,
            max(0, len(messages) - head_end - _MINIMUM_MESSAGES_TO_COMPACT),
        )
        if minimum_tail <= 0 and token_budget <= 0:
            return len(messages)

        accumulated_tokens: int = 0
        boundary: int = len(messages)
        if token_budget > 0:
            index: int
            for index in range(len(messages) - 1, head_end - 1, -1):
                message: Mapping[str, object] = messages[index]
                message_tokens: int = self._message_token_estimate(message)
                keep_count: int = len(messages) - index
                if (
                    accumulated_tokens + message_tokens > token_budget
                    and keep_count >= minimum_tail
                ):
                    break
                accumulated_tokens += message_tokens
                boundary = index

        fallback_boundary: int = (
            max(head_end + 1, len(messages) - minimum_tail)
            if minimum_tail > 0
            else len(messages)
        )
        if boundary <= head_end:
            boundary = head_end + 1 if minimum_tail <= 0 else fallback_boundary
        if minimum_tail > 0 and boundary > fallback_boundary:
            boundary = fallback_boundary
        return self._align_boundary_backward(messages, boundary)

    def _select_compaction_window(
        self,
        messages: Sequence[Mapping[str, object]],
    ) -> CompactionWindow | None:
        copied_messages: list[ChatMessage] = _copy_messages(messages)
        minimum_message_count: int = (
            self.protect_first_n + self.minimum_tail_messages + _MINIMUM_MESSAGES_TO_COMPACT
        )
        if len(copied_messages) <= minimum_message_count:
            return None

        head_end: int = self._align_boundary_forward(copied_messages, self.protect_first_n)
        tail_start: int = self._find_tail_start(copied_messages, head_end)
        if tail_start <= head_end:
            return None

        middle: list[ChatMessage] = copied_messages[head_end:tail_start]
        if len(middle) < _MINIMUM_MESSAGES_TO_COMPACT:
            return None

        return CompactionWindow(
            head=copied_messages[:head_end],
            middle=middle,
            tail=copied_messages[tail_start:],
            compressed_count=len(middle),
        )

    def _fallback_checkpoint(self, dropped_turn_count: int) -> str:
        summary_body: str = self._fallback_checkpoint_body(dropped_turn_count)
        self.last_checkpoint_focus_topic = None
        self.last_checkpoint_summary = summary_body.strip()
        self._previous_summary = self.last_checkpoint_summary
        return self._with_summary_prefix(self.last_checkpoint_summary)

    def _checkpoint_role_and_merge(
        self,
        window: CompactionWindow,
    ) -> tuple[str, bool]:
        if not window.tail:
            return ("assistant", False)
        last_head_role: str = _message_role(window.head[-1]) if window.head else "user"
        first_tail_role: str = _message_role(window.tail[0])
        summary_role: str = "user" if last_head_role in {"assistant", "tool"} else "assistant"
        if summary_role == first_tail_role:
            flipped_role: str = "assistant" if summary_role == "user" else "user"
            if flipped_role != last_head_role:
                summary_role = flipped_role
            else:
                return (summary_role, True)
        return (summary_role, False)

    def _sanitize_tool_pairs(
        self,
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        surviving_call_ids: set[str] = set()
        message: ChatMessage
        for message in messages:
            if _message_role(message) != "assistant":
                continue
            tool_call: ToolCallPayload
            for tool_call in _tool_calls(message):
                call_id: str = _tool_call_id(tool_call)
                if call_id:
                    surviving_call_ids.add(call_id)

        result_call_ids: set[str] = set()
        for message in messages:
            if _message_role(message) != "tool":
                continue
            tool_call_id_value: str = string_or_empty(message.get("tool_call_id")).strip()
            if tool_call_id_value:
                result_call_ids.add(tool_call_id_value)

        orphaned_results: set[str] = result_call_ids - surviving_call_ids
        sanitized_messages: list[ChatMessage] = []
        for message in messages:
            if (
                _message_role(message) == "tool"
                and string_or_empty(message.get("tool_call_id")).strip() in orphaned_results
            ):
                continue
            sanitized_messages.append(message)

        missing_results: set[str] = surviving_call_ids - result_call_ids
        if not missing_results:
            return sanitized_messages

        patched_messages: list[ChatMessage] = []
        for message in sanitized_messages:
            patched_messages.append(message)
            if _message_role(message) != "assistant":
                continue
            tool_call: ToolCallPayload
            for tool_call in _tool_calls(message):
                call_id = _tool_call_id(tool_call)
                if call_id in missing_results:
                    patched_messages.append(
                        {
                            "role": "tool",
                            "content": "[Result from earlier conversation — see context summary above]",
                            "tool_call_id": call_id,
                        }
                    )
        return patched_messages

    def _assemble_compacted_messages(
        self,
        window: CompactionWindow,
        checkpoint_message: str,
    ) -> list[ChatMessage]:
        checkpoint_role, merge_into_tail = self._checkpoint_role_and_merge(window)
        compacted: list[ChatMessage] = _copy_messages(window.head)
        if not merge_into_tail:
            compacted.append({"role": checkpoint_role, "content": checkpoint_message})

        index: int
        tail_message: ChatMessage
        for index, tail_message in enumerate(_copy_messages(window.tail)):
            if merge_into_tail and index == 0:
                original_content: str = _message_content(tail_message)
                tail_message["content"] = (
                    checkpoint_message
                    + f"\n\n{_CHECKPOINT_SEPARATOR}\n\n"
                    + original_content
                )
            compacted.append(tail_message)
        return compacted

    def compress(
        self,
        messages: list[dict[str, object]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> list[dict[str, object]]:
        del current_tokens
        window: CompactionWindow | None = self._select_compaction_window(messages)
        if window is None:
            return _runtime_messages(_copy_messages(messages))

        checkpoint_message: str | None = self._generate_summary(window.middle, focus_topic)
        if checkpoint_message is None:
            checkpoint_message = self._fallback_checkpoint(window.compressed_count)

        compacted_messages: list[ChatMessage] = self._assemble_compacted_messages(
            window,
            checkpoint_message,
        )
        sanitized_messages: list[ChatMessage] = self._sanitize_tool_pairs(compacted_messages)
        self.compression_count += 1
        self.last_checkpoint_focus_topic = focus_topic
        self.last_checkpoint_summary = strip_summary_prefix(checkpoint_message)
        self._previous_summary = self.last_checkpoint_summary
        compacted_estimate: int = estimate_messages_tokens_rough(
            _runtime_messages(sanitized_messages)
        )
        self.last_prompt_tokens = compacted_estimate
        self.last_completion_tokens = 0
        self.last_total_tokens = compacted_estimate
        if self._compaction_callback and self.last_checkpoint_summary:
            self._compaction_callback(
                {
                    "focus_topic": focus_topic,
                    "summary": self.last_checkpoint_summary,
                }
            )
        return _runtime_messages(sanitized_messages)


def register(ctx: object) -> None:
    register_context_engine: object = getattr(
        ctx,
        "register_context_engine",
        None,
    )
    if not callable(register_context_engine):
        raise TypeError("Plugin registration context is unavailable")
    register_context_engine(OperationalCheckpointCompressor())
    install_plugin_sidecar()
