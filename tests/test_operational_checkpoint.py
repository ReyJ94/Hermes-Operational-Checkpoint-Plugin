from __future__ import annotations

import importlib
import importlib.util
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import yaml

import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
PLUGIN_ENTRYPOINT: Path = ROOT / "__init__.py"
PluginModules = tuple[ModuleType, ModuleType, ModuleType, ModuleType]


def checkpoint_summary(objective: str = "Continue the task") -> str:
    return "\n\n".join(
        [
            f"1. Objective\n- [Observed] {objective}",
            "2. Explicit user instructions / prohibitions / scope boundaries\nNone.",
            "3. Operational state\nNone.",
            "4. Active working set\nNone.",
            "5. Discoveries / evidence\nNone.",
            "6. Settled decisions / rejected alternatives\nNone.",
            "7. Transferable patterns learned this run\nNone.",
            "8. Assumptions / uncertainties / blockers\nNone.",
            "9. Execution status\n- Accomplished: None.\n- In progress: None.\n- Remaining: None.",
            "10. Action frontier\n- [Observed] Run the next probe.",
            "11. Critical invariants / regression risks\nNone.",
        ]
    )


def load_plugin_entrypoint() -> ModuleType:
    module_name = "operational_checkpoint_plugin_entrypoint"
    if module_name in sys.modules:
        cached_module: ModuleType = sys.modules[module_name]
        return cached_module

    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_ENTRYPOINT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create plugin entrypoint module spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_modules() -> PluginModules:
    entrypoint_module = load_plugin_entrypoint()
    compressor_module = importlib.import_module("operational_checkpoint.compressor")
    helpers_module = importlib.import_module("operational_checkpoint.helpers")
    sidecar_module = importlib.import_module("operational_checkpoint.sidecar")
    return entrypoint_module, compressor_module, helpers_module, sidecar_module


def test_ensure_plugin_activation_writes_yaml_list_and_context_engine(
    tmp_path: Path,
) -> None:
    from operational_checkpoint.activation import ensure_plugin_activation

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "plugins": {"enabled": ["disk-cleanup"]},
                "context": {"engine": "compressor"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    changed = ensure_plugin_activation(config_path)

    assert changed is True
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["plugins"]["enabled"] == ["disk-cleanup", "operational_checkpoint"]
    assert config["context"]["engine"] == "operational_checkpoint"


def test_ensure_plugin_activation_replaces_stringified_enabled_list(
    tmp_path: Path,
) -> None:
    from operational_checkpoint.activation import ensure_plugin_activation

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "plugins": {"enabled": '["operational_checkpoint"]'},
                "context": {"engine": "operational_checkpoint"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    changed = ensure_plugin_activation(config_path)

    assert changed is True
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["plugins"]["enabled"] == ["operational_checkpoint"]


def test_ensure_plugin_activation_is_idempotent(
    tmp_path: Path,
) -> None:
    from operational_checkpoint.activation import ensure_plugin_activation

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "plugins": {"enabled": ["operational_checkpoint"]},
                "context": {"engine": "operational_checkpoint"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    changed = ensure_plugin_activation(config_path)

    assert changed is False


def test_registers_context_engine(
    plugin_modules: PluginModules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package, _compressor_module, _helpers_module, _sidecar_module = plugin_modules
    install_calls: list[str] = []

    class Collector:
        def __init__(self) -> None:
            self.engine: object | None = None

        def register_context_engine(self, engine: object) -> None:
            self.engine = engine

    monkeypatch.setitem(
        package.register.__globals__,
        "install_plugin_sidecar",
        lambda: install_calls.append("installed"),
    )

    collector = Collector()
    package.register(collector)

    engine = collector.engine
    assert engine is not None
    assert getattr(engine, "name") == "operational_checkpoint"
    assert install_calls == ["installed"]


def test_install_plugin_sidecar_defers_when_cli_is_partially_initialized(
    plugin_modules: PluginModules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules
    partial_cli_module = ModuleType("cli")
    partial_run_agent_module = ModuleType("run_agent")
    scheduled_calls: list[str] = []

    monkeypatch.setitem(sys.modules, "cli", partial_cli_module)
    monkeypatch.setitem(sys.modules, "run_agent", partial_run_agent_module)
    monkeypatch.setattr(
        sidecar_module,
        "_schedule_deferred_install",
        lambda: scheduled_calls.append("scheduled"),
    )
    monkeypatch.setattr(
        sidecar_module,
        "install_runtime_bridge",
        lambda **_kwargs: pytest.fail("bridge should not install before HermesCLI exists"),
    )

    sidecar_module.install_plugin_sidecar()

    assert scheduled_calls == ["scheduled"]


def test_install_plugin_sidecar_defers_when_discovered_before_cli_import(
    plugin_modules: PluginModules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules
    partial_run_agent_module = ModuleType("run_agent")
    scheduled_calls: list[str] = []

    monkeypatch.delitem(sys.modules, "cli", raising=False)
    monkeypatch.setitem(sys.modules, "run_agent", partial_run_agent_module)
    monkeypatch.setattr(
        sidecar_module,
        "_schedule_deferred_install",
        lambda: scheduled_calls.append("scheduled"),
    )
    monkeypatch.setattr(
        sidecar_module,
        "install_runtime_bridge",
        lambda **_kwargs: pytest.fail("CLI bridge should not install before HermesCLI exists"),
    )

    sidecar_module.install_plugin_sidecar()

    assert scheduled_calls == ["scheduled"]


def test_install_plugin_sidecar_patches_agent_when_tui_imports_run_agent_without_cli(
    plugin_modules: PluginModules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules
    run_agent_module = ModuleType("run_agent")
    installed_agents: list[type[object]] = []
    scheduled_calls: list[str] = []

    class FakeAgent:
        def _compress_context(
            self,
            *args: object,
            **kwargs: object,
        ) -> tuple[object, object]:
            del args, kwargs
            return [], ""

    run_agent_module.AIAgent = FakeAgent
    monkeypatch.delitem(sys.modules, "cli", raising=False)
    monkeypatch.setitem(sys.modules, "run_agent", run_agent_module)
    monkeypatch.setattr(
        sidecar_module,
        "install_agent_runtime_bridge",
        lambda *, agent_class: installed_agents.append(agent_class),
    )
    monkeypatch.setattr(
        sidecar_module,
        "install_runtime_bridge",
        lambda **_kwargs: pytest.fail("CLI bridge should not install before HermesCLI exists"),
    )
    monkeypatch.setattr(
        sidecar_module,
        "_schedule_deferred_install",
        lambda: scheduled_calls.append("scheduled"),
    )

    sidecar_module.install_plugin_sidecar()
    assert installed_agents == [FakeAgent]
    assert scheduled_calls == ["scheduled"]


def test_agent_runtime_bridge_refreshes_tui_usage_after_response_usage_update(
    plugin_modules: PluginModules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules
    tui_module = ModuleType("tui_gateway.server")
    emitted: list[tuple[str, str, dict[str, object]]] = []

    class FakeEngine:
        def __init__(self) -> None:
            self.bound_session_id = ""
            self.compression_count = 0
            self.context_length = 400_000
            self.hermes_home = None
            self.last_checkpoint_summary = ""
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.threshold_tokens = 350_000

        def update_from_response(self, usage: dict[str, object]) -> None:
            self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            del callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            return False

        def persist_usage_snapshot(self) -> None:
            return None

        def compress(
            self,
            messages: list[dict[str, object]],
            current_tokens: int | None = None,
            focus_topic: str | None = None,
        ) -> list[dict[str, object]]:
            del current_tokens, focus_topic
            return messages

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self.platform = "tui"
            self.session_id = "session-live"

        def _compress_context(
            self,
            messages: list[dict[str, object]],
            system_message: str,
            **runtime_kwargs: object,
        ) -> tuple[list[dict[str, object]], str]:
            del runtime_kwargs
            return messages, system_message

    def emit(event: str, sid: str, payload: dict[str, object]) -> None:
        emitted.append((event, sid, payload))

    def session_info(agent: object) -> dict[str, object]:
        engine = getattr(agent, "context_compressor")
        return {
            "usage": {
                "context_used": engine.last_prompt_tokens,
                "context_max": engine.context_length,
                "context_percent": round(engine.last_prompt_tokens / engine.context_length * 100),
            }
        }

    tui_module._emit = emit
    tui_module._session_info = session_info
    tui_module._sessions = {}
    monkeypatch.setitem(sys.modules, "tui_gateway.server", tui_module)
    monkeypatch.setattr(
        sidecar_module,
        "estimate_request_tokens_rough",
        lambda messages, *, system_prompt="", tools=None: len(messages) * 100 + len(system_prompt) + (7 if tools else 0),
    )

    sidecar_module.install_agent_runtime_bridge(agent_class=FakeAgent)
    agent = FakeAgent()
    agent._cached_system_prompt = "system"
    agent.tools = [{"type": "function", "function": {"name": "noop"}}]
    tui_module._sessions = {
        "sid-live": {
            "agent": agent,
            "history": [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        }
    }

    agent.context_compressor.update_from_response(
        {"prompt_tokens": 252_000, "completion_tokens": 1_000, "total_tokens": 253_000}
    )

    expected_context_used = 2 * 100 + len("system") + 7
    assert agent.context_compressor.bound_session_id == "session-live"
    assert agent.context_compressor.last_prompt_tokens == expected_context_used
    assert agent.context_compressor.last_completion_tokens == 0
    assert agent.context_compressor.last_total_tokens == expected_context_used
    assert emitted == [
        (
            "session.info",
            "sid-live",
            {"usage": {"context_used": expected_context_used, "context_max": 400_000, "context_percent": 0}},
        )
    ]



def test_tui_usage_refresh_prefers_active_compacted_messages_over_visible_history(
    plugin_modules: PluginModules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules
    tui_module = ModuleType("tui_gateway.server")
    emitted: list[tuple[str, str, dict[str, object]]] = []

    class FakeEngine:
        def __init__(self) -> None:
            self.bound_session_id = ""
            self.compression_count = 1
            self.context_length = 400_000
            self.hermes_home = None
            self.last_checkpoint_summary = "summary"
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.threshold_tokens = 350_000

        def update_from_response(self, usage: dict[str, object]) -> None:
            self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            del callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            return False

        def persist_usage_snapshot(self) -> None:
            return None

        def compress(
            self,
            messages: list[dict[str, object]],
            current_tokens: int | None = None,
            focus_topic: str | None = None,
        ) -> list[dict[str, object]]:
            del current_tokens, focus_topic
            return messages

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self.platform = "tui"
            self.session_id = "session-live"
            self._session_messages = [{"role": "assistant", "content": "checkpoint"}]

        def _compress_context(
            self,
            messages: list[dict[str, object]],
            system_message: str,
            **runtime_kwargs: object,
        ) -> tuple[list[dict[str, object]], str]:
            del runtime_kwargs
            return messages, system_message

    def emit(event: str, sid: str, payload: dict[str, object]) -> None:
        emitted.append((event, sid, payload))

    def session_info(agent: object) -> dict[str, object]:
        engine = getattr(agent, "context_compressor")
        return {"usage": {"context_used": engine.last_prompt_tokens}}

    tui_module._emit = emit
    tui_module._session_info = session_info
    tui_module._sessions = {}
    monkeypatch.setitem(sys.modules, "tui_gateway.server", tui_module)
    monkeypatch.setattr(
        sidecar_module,
        "estimate_request_tokens_rough",
        lambda messages, *, system_prompt="", tools=None: len(messages) * 100,
    )

    sidecar_module.install_agent_runtime_bridge(agent_class=FakeAgent)
    agent = FakeAgent()
    tui_module._sessions = {
        "sid-live": {
            "agent": agent,
            "history": [{"role": "user", "content": str(index)} for index in range(20)],
        }
    }

    agent.context_compressor.update_from_response(
        {"prompt_tokens": 252_000, "completion_tokens": 1_000, "total_tokens": 253_000}
    )

    assert agent.context_compressor.last_prompt_tokens == 100
    assert emitted == [("session.info", "sid-live", {"usage": {"context_used": 100}})]



def test_tui_prompt_submit_uses_compacted_active_history_without_replacing_visible_history(
    plugin_modules: PluginModules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package, _compressor_module, helpers_module, sidecar_module = plugin_modules
    tui_module = ModuleType("tui_gateway.server")
    emitted: list[tuple[str, str, dict[str, object]]] = []
    seen_histories: list[list[dict[str, object]]] = []

    class FakeEngine:
        def __init__(self) -> None:
            self.bound_session_id = ""
            self.compression_count = 1
            self.context_length = 400_000
            self.hermes_home = None
            self.last_checkpoint_summary = "summary"
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.threshold_tokens = 350_000

        def update_from_response(self, usage: dict[str, object]) -> None:
            self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            del callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            return False

        def persist_usage_snapshot(self) -> None:
            return None

        def compress(
            self,
            messages: list[dict[str, object]],
            current_tokens: int | None = None,
            focus_topic: str | None = None,
        ) -> list[dict[str, object]]:
            del current_tokens, focus_topic
            return messages

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self.platform = "tui"
            self.session_id = "session-live"
            self._cached_system_prompt = "system"
            self.tools = []

        def _compress_context(
            self,
            messages: list[dict[str, object]],
            system_message: str,
            **runtime_kwargs: object,
        ) -> tuple[list[dict[str, object]], str]:
            del runtime_kwargs
            return messages, system_message

        def run_conversation(
            self,
            user_message: str,
            system_message: str | None = None,
            conversation_history: list[dict[str, object]] | None = None,
            **kwargs: object,
        ) -> dict[str, object]:
            del system_message, kwargs
            seen = list(conversation_history or [])
            seen_histories.append(seen)
            return {
                "final_response": "ok",
                "messages": seen
                + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "ok"},
                ],
            }

    def emit(event: str, sid: str, payload: dict[str, object]) -> None:
        emitted.append((event, sid, payload))

    def session_info(agent: object) -> dict[str, object]:
        engine = getattr(agent, "context_compressor")
        return {"usage": {"context_used": engine.last_prompt_tokens}}

    tui_module._emit = emit
    tui_module._session_info = session_info
    tui_module._sessions = {}
    monkeypatch.setitem(sys.modules, "tui_gateway.server", tui_module)
    monkeypatch.setattr(
        sidecar_module,
        "estimate_request_tokens_rough",
        lambda messages, *, system_prompt="", tools=None: len(messages) * 100 + len(system_prompt),
    )

    sidecar_module.install_agent_runtime_bridge(agent_class=FakeAgent)
    agent = FakeAgent()
    raw_history = [{"role": "user", "content": f"raw-{index}"} for index in range(12)]
    record = helpers_module.PersistedCompactionState(
        compacted_messages=[{"role": "assistant", "content": "checkpoint"}],
        compression_count=1,
        focus_topic=None,
        raw_message_count=20,
        summary="summary",
        tokens_after=100,
        tokens_before=1000,
        updated_at=123.0,
    )
    sidecar_module._set_active_compaction_state(agent=agent, record=record)
    tui_module._sessions = {
        "sid-live": {
            "agent": agent,
            "history": raw_history,
        }
    }

    result = agent.run_conversation("new", conversation_history=list(raw_history))

    assert seen_histories == [[{"role": "assistant", "content": "checkpoint"}]]
    assert result["messages"] == raw_history + [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "ok"},
    ]
    assert agent._session_messages == [
        {"role": "assistant", "content": "checkpoint"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "ok"},
    ]
    assert agent.context_compressor.last_prompt_tokens == 3 * 100 + len("system")
    assert emitted == [
        ("session.info", "sid-live", {"usage": {"context_used": 206}}),
        ("session.info", "sid-live", {"usage": {"context_used": 306}}),
    ]



def test_tui_hydration_uses_compacted_fallback_when_checkpoint_is_ahead_of_history(
    plugin_modules: PluginModules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package, _compressor_module, helpers_module, sidecar_module = plugin_modules

    class FakeEngine:
        def __init__(self) -> None:
            self.compression_count = 1
            self.context_length = 400_000
            self.hermes_home = None
            self.last_checkpoint_summary = "summary"
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.threshold_tokens = 350_000

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            del callback

        def persist_usage_snapshot(self) -> None:
            self.persisted = True

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self.session_id = "session-live"

    agent = FakeAgent()
    raw_history = [
        {"role": "user", "content": "real latest request"},
        {"role": "assistant", "content": "real answer"},
    ]
    session: dict[str, object] = {
        "agent": agent,
        "session_key": "session-live",
        "history": list(raw_history),
    }
    compacted = [{"role": "assistant", "content": "checkpoint only"}]
    record = helpers_module.PersistedCompactionState(
        compacted_messages=compacted,
        compression_count=7,
        focus_topic=None,
        raw_message_count=200,
        summary="checkpoint only",
        tokens_after=24_251,
        tokens_before=36_565,
        updated_at=1.0,
    )
    monkeypatch.setattr(
        sidecar_module,
        "_load_persisted_compaction_state",
        lambda *, session_id, hermes_home: record,
    )

    monkeypatch.setattr(sidecar_module, "_estimate_request_tokens", lambda **kwargs: 24_251)

    sidecar_module._hydrate_tui_session_history_from_plugin_state(session)

    assert session["history"] == compacted
    assert agent._session_messages == compacted
    assert agent.context_compressor.last_prompt_tokens == 24_251
    assert agent.context_compressor.last_total_tokens == 24_251
    assert agent.context_compressor.persisted is True



def test_install_plugin_sidecar_installs_when_targets_are_ready(
    plugin_modules: PluginModules,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules

    class FakeCLI:
        session_id = "session-1"

        def _init_agent(self, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            return True

        def _manual_compress(self, cmd_original: str = "") -> None:
            del cmd_original

    class FakeAgent:
        def _compress_context(
            self,
            *args: object,
            **kwargs: object,
        ) -> tuple[object, object]:
            del args, kwargs
            return [], ""

    cli_module = ModuleType("cli")
    cli_module.HermesCLI = FakeCLI
    run_agent_module = ModuleType("run_agent")
    run_agent_module.AIAgent = FakeAgent
    captured: dict[str, object] = {}

    monkeypatch.setitem(sys.modules, "cli", cli_module)
    monkeypatch.setitem(sys.modules, "run_agent", run_agent_module)
    monkeypatch.setattr(
        sidecar_module,
        "_schedule_deferred_install",
        lambda: pytest.fail("deferred install should not be needed"),
    )

    def fake_install_runtime_bridge(
        *,
        cli_class: type[object],
        agent_class: type[object],
    ) -> None:
        captured["cli_class"] = cli_class
        captured["agent_class"] = agent_class

    monkeypatch.setattr(
        sidecar_module,
        "install_runtime_bridge",
        fake_install_runtime_bridge,
    )

    sidecar_module.install_plugin_sidecar()

    assert captured == {
        "cli_class": FakeCLI,
        "agent_class": FakeAgent,
    }


def test_prompt_uses_operational_checkpoint_sections(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "medium",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    captured: dict[str, object] = {}

    def fake_call_llm(
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
        del provider, model, base_url, api_key, max_tokens, main_runtime
        captured["task"] = task
        captured["prompt"] = messages[0]["content"]
        captured["extra_body"] = extra_body
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=checkpoint_summary("Continue the task")
                    )
                )
            ]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    result = engine._generate_summary(
        [{"role": "user", "content": "Investigate the adapter failure"}],
        focus_topic="adapter failure",
    )

    prompt = str(captured["prompt"])
    assert captured["task"] == "compression"
    assert captured["extra_body"] == {"reasoning": {"effort": "medium"}}
    assert "Operational state" in prompt
    assert "Transferable patterns learned this run" in prompt
    assert "Action frontier" in prompt
    assert "Critical invariants / regression risks" in prompt
    assert "FOCUS TOPIC: adapter failure" in prompt
    assert result is not None
    assert result.startswith(compressor_module.SUMMARY_PREFIX)


def test_generate_summary_rejects_malformed_six_section_checkpoint_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4-mini",
            "provider": "openai-codex",
            "reasoning_effort": "medium",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )
    malformed = (
        "1. Objective\n- [Observed] Continue\n\n"
        "2. Explicit user instructions / prohibitions / scope boundaries\nNone.\n\n"
        "3. Operational state\nNone.\n\n"
        "4. Active working set\nNone.\n\n"
        "5. Discoveries / evidence\nNone.\n\n"
        "6. Settled decisions / rejected alternatives\n"
        "- [Settled? invalid label prohibited] None.\n"
        "Need only allowed labels. Must avoid \"Settled?\" label. Rewrite section with allowed labels."
    )
    responses = iter([malformed, checkpoint_summary("valid retry")])
    attempts: list[str] = []

    def fake_call_llm(
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
        del provider, model, base_url, api_key, task, messages, max_tokens, extra_body, main_runtime
        content = next(responses)
        attempts.append(content)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    result = engine._generate_summary([{"role": "user", "content": "compress me"}])

    assert result is not None
    assert len(attempts) == 2
    assert "7. Transferable patterns learned this run" in result
    assert "[Settled? invalid label prohibited]" not in result
    assert engine.last_checkpoint_summary.startswith("1. Objective\n- [Observed] valid retry")


def test_fallback_checkpoint_preserves_canonical_eleven_section_shape(
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    body = compressor_module.OperationalCheckpointCompressor._fallback_checkpoint_body(7)

    for index, section_title in enumerate(compressor_module._REQUIRED_CHECKPOINT_SECTIONS, start=1):
        assert f"{index}. {section_title}" in body
    assert not compressor_module.OperationalCheckpointCompressor._summary_validation_errors(body)
    assert "7 earlier turns were compacted" in body


def test_runtime_defaults_drive_threshold_shape(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    engine = compressor_module.OperationalCheckpointCompressor()

    assert engine.name == "operational_checkpoint"
    assert engine.context_length == 200_000
    assert engine.threshold_tokens == 175_000
    assert engine.protect_first_n == 2
    assert engine.protect_last_n == 3
    assert engine.tail_token_budget == 18_000


def test_operational_checkpoint_is_a_direct_context_engine(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules
    from agent.context_engine import ContextEngine

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    engine = compressor_module.OperationalCheckpointCompressor()

    assert isinstance(engine, ContextEngine)


def test_compress_uses_plugin_owned_checkpoint_assembly_and_updates_previous_summary(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 900,
            "base_url": "",
            "config_context_length": 1_000,
            "context_limit_tokens": 1_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 2,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 80,
        },
    )

    prompts: list[str] = []
    summaries = iter(
        [
            checkpoint_summary("first checkpoint"),
            checkpoint_summary("updated checkpoint"),
        ]
    )

    def fake_call_llm(
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
        del provider, model, base_url, api_key, task, max_tokens, extra_body, main_runtime
        prompts.append(messages[0]["content"])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=next(summaries)
                    )
                )
            ]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    initial_history = [
        {"role": "user", "content": "u0 " + ("x" * 280)},
        {"role": "assistant", "content": "a0 " + ("y" * 280)},
        {"role": "user", "content": "u1 " + ("x" * 280)},
        {"role": "assistant", "content": "a1 " + ("y" * 280)},
        {"role": "user", "content": "u2 " + ("x" * 280)},
        {"role": "assistant", "content": "a2 " + ("y" * 280)},
    ]

    compressed_once = engine.compress(initial_history, focus_topic="adapter seam")

    checkpoint_messages_once = [
        message for message in compressed_once
        if compressor_module.SUMMARY_PREFIX in str(message.get("content", ""))
    ]
    assert engine.compression_count == 1
    assert len(checkpoint_messages_once) == 1
    assert "PREVIOUS CHECKPOINT" not in prompts[0]
    assert "FOCUS TOPIC: adapter seam" in prompts[0]

    second_history = list(compressed_once) + [
        {"role": "user", "content": "u3 " + ("x" * 280)},
        {"role": "assistant", "content": "a3 " + ("y" * 280)},
        {"role": "user", "content": "u4 " + ("x" * 280)},
        {"role": "assistant", "content": "a4 " + ("y" * 280)},
    ]

    compressed_twice = engine.compress(second_history)

    checkpoint_messages_twice = [
        message for message in compressed_twice
        if compressor_module.SUMMARY_PREFIX in str(message.get("content", ""))
    ]
    assert engine.compression_count == 2
    assert len(checkpoint_messages_twice) == 1
    assert "PREVIOUS CHECKPOINT:\n1. Objective\n- [Observed] first checkpoint" in prompts[1]
    assert "NEW TURNS TO INCORPORATE:" in prompts[1]
    assert engine.last_checkpoint_summary.startswith("1. Objective\n- [Observed] updated checkpoint")



def test_compress_preserves_latest_turn_even_with_zero_configured_tail(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 900,
            "base_url": "",
            "config_context_length": 1_000,
            "context_limit_tokens": 1_000,
            "head_preserve_messages": 0,
            "minimum_tail_messages": 0,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 0,
        },
    )
    monkeypatch.setattr(
        compressor_module.OperationalCheckpointCompressor,
        "_generate_summary",
        lambda self, turns_to_summarize, focus_topic=None: self._with_summary_prefix(
            checkpoint_summary("preserve current request")
        ),
    )

    engine = compressor_module.OperationalCheckpointCompressor()
    latest_user = {"role": "user", "content": "CURRENT REQUEST: answer this"}
    compacted = engine.compress(
        [
            {"role": "user", "content": "old user " + ("x" * 280)},
            {"role": "assistant", "content": "old assistant " + ("y" * 280)},
            latest_user,
        ]
    )

    assert compacted[-1] == latest_user
    assert any(
        compressor_module.SUMMARY_PREFIX in str(message.get("content", ""))
        for message in compacted[:-1]
    )



def test_compress_inserts_plugin_owned_fallback_checkpoint_when_summary_fails(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 900,
            "base_url": "",
            "config_context_length": 1_000,
            "context_limit_tokens": 1_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 2,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 80,
        },
    )

    call_attempts: list[int] = []

    def fake_call_llm(
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
        del provider, model, base_url, api_key, task, messages, max_tokens, extra_body, main_runtime
        call_attempts.append(1)
        raise RuntimeError("summary unavailable")

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    compacted = engine.compress(
        [
            {"role": "user", "content": "u0 " + ("x" * 280)},
            {"role": "assistant", "content": "a0 " + ("y" * 280)},
            {"role": "user", "content": "u1 " + ("x" * 280)},
            {"role": "assistant", "content": "a1 " + ("y" * 280)},
            {"role": "user", "content": "u2 " + ("x" * 280)},
            {"role": "assistant", "content": "a2 " + ("y" * 280)},
        ]
    )

    checkpoint_messages = [
        message for message in compacted
        if compressor_module.SUMMARY_PREFIX in str(message.get("content", ""))
    ]
    assert engine.compression_count == 1
    assert len(checkpoint_messages) == 1
    assert "Summary generation was unavailable" in str(
        checkpoint_messages[0].get("content", "")
    )
    assert "Summary generation was unavailable" in engine.last_checkpoint_summary
    assert len(call_attempts) == 3


def test_compress_retries_empty_summary_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 900,
            "base_url": "",
            "config_context_length": 1_000,
            "context_limit_tokens": 1_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 2,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 80,
        },
    )

    call_attempts: list[int] = []

    def fake_call_llm(
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
        del provider, model, base_url, api_key, task, messages, max_tokens, extra_body, main_runtime
        call_attempts.append(1)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    compacted = engine.compress(
        [
            {"role": "user", "content": "u0 " + ("x" * 280)},
            {"role": "assistant", "content": "a0 " + ("y" * 280)},
            {"role": "user", "content": "u1 " + ("x" * 280)},
            {"role": "assistant", "content": "a1 " + ("y" * 280)},
            {"role": "user", "content": "u2 " + ("x" * 280)},
            {"role": "assistant", "content": "a2 " + ("y" * 280)},
        ]
    )

    checkpoint_messages = [
        message for message in compacted
        if compressor_module.SUMMARY_PREFIX in str(message.get("content", ""))
    ]
    assert len(call_attempts) == 3
    assert len(checkpoint_messages) == 1
    assert "Summary generation was unavailable" in str(
        checkpoint_messages[0].get("content", "")
    )


def test_sidecar_bridge_emits_auto_status_and_records_artifact(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules
    stored_states: dict[str, object] = {}

    monkeypatch.setattr(
        sidecar_module,
        "load_operational_checkpoint_cli_config",
        lambda: {
            "emit_compaction_status": True,
            "show_summary_preview": True,
            "summary_preview_chars": 48,
        },
    )
    monkeypatch.setattr(
        sidecar_module,
        "load_compaction_states",
        lambda hermes_home=None: dict(stored_states),
    )

    def save_states(states: dict[str, object], hermes_home: object | None = None) -> None:
        del hermes_home
        stored_states.clear()
        stored_states.update(states)

    monkeypatch.setattr(sidecar_module, "save_compaction_states", save_states)

    class FakeEngine:
        def __init__(self) -> None:
            self.callback: Callable[[dict[str, str | None]], None] | None = None
            self.compression_count = 0
            self.last_completion_tokens = 0
            self.last_checkpoint_summary = ""
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.threshold_tokens = 350_000

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            self.callback = callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            return True

        def persist_usage_snapshot(self) -> None:
            return None

        def compress(
            self,
            messages: list[dict[str, object]],
            current_tokens: int | None = None,
            focus_topic: str | None = None,
        ) -> list[dict[str, object]]:
            del messages, current_tokens
            self.compression_count += 1
            self.last_checkpoint_summary = (
                "Goal keep moving. Run the next discriminating probe."
            )
            if self.callback is not None:
                self.callback(
                    {
                        "focus_topic": focus_topic,
                        "summary": self.last_checkpoint_summary,
                    }
                )
            return [{"role": "user", "content": "checkpoint"}]

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self._cached_system_prompt = ""
            self._context_pressure_warned_at = 0.0
            self._session_db = None
            self.printed: list[str] = []
            self.session_id = "session_123"
            self.tools: list[dict[str, object]] = []

        def _safe_print(self, line: str) -> None:
            self.printed.append(line)

        def flush_memories(
            self,
            messages: list[dict[str, object]],
            min_turns: int = 0,
        ) -> None:
            del messages, min_turns

        def _invalidate_system_prompt(self) -> None:
            self._cached_system_prompt = ""

        def _build_system_prompt(self, system_message: str) -> str:
            return system_message

        def _compress_context(self, *args: object, **kwargs: object) -> tuple[list[dict[str, object]], str]:
            del args, kwargs
            raise AssertionError("plugin-owned compaction should bypass the original _compress_context")

    class FakeCLI:
        def __init__(self) -> None:
            self.agent = FakeAgent()

        def _init_agent(self, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            return True

        def _manual_compress(self, cmd_original: str = "") -> None:
            del cmd_original
            self.agent._compress_context(
                [{"role": "user", "content": "manual"}],
                "",
                approx_tokens=90_000,
                focus_topic="manual focus",
            )

    sidecar_module.install_runtime_bridge(cli_class=FakeCLI, agent_class=FakeAgent)

    cli = FakeCLI()
    assert cli._init_agent() is True
    assert cli.agent.context_compressor.callback is not None

    compressed, _system_prompt = cli.agent._compress_context(
        [{"role": "user", "content": "auto"}] * 8,
        "",
        approx_tokens=123_456,
        focus_topic="adapter seam",
    )
    expected_tokens_before = sidecar_module.estimate_request_tokens_rough(
        [{"role": "user", "content": "auto"}] * 8,
        system_prompt="",
        tools=[],
    )
    expected_tokens_after = sidecar_module.estimate_request_tokens_rough(
        compressed,
        system_prompt="",
        tools=[],
    )

    assert compressed == [{"role": "user", "content": "checkpoint"}]
    assert cli.agent.printed[0] == (
        f"Operational Checkpoint: auto-compacting ~{expected_tokens_before:,} / 350,000 "
        'tokens, focus: "adapter seam"...'
    )
    assert cli.agent.printed[1] == (
        "  Operational Checkpoint reduced active context budget: "
        f"~{expected_tokens_before:,} → ~{expected_tokens_after:,} tokens"
    )
    assert cli.agent.printed[2].startswith("     checkpoint: Goal keep moving.")
    artifact = sidecar_module.latest_compaction_artifact_for_cli(cli)
    assert artifact is not None
    assert artifact["focus_topic"] == "adapter seam"
    assert artifact["trigger"] == "auto"
    assert artifact["message_count_before"] == 8
    assert artifact["message_count_after"] == 1
    assert artifact["tokens_before"] == expected_tokens_before
    assert artifact["tokens_after"] == expected_tokens_after
    stored_record = stored_states["session_123"]
    assert getattr(stored_record, "raw_message_count") == 8
    assert getattr(stored_record, "raw_cursor_message_count") == 8
    assert getattr(stored_record, "raw_cursor_message_hash")
    assert ":1:8:" in getattr(stored_record, "checkpoint_id")
    assert getattr(stored_record, "tokens_after") == expected_tokens_after

    before_manual = list(cli.agent.printed)
    cli._manual_compress()
    assert cli.agent.printed == before_manual
    manual_artifact = sidecar_module.latest_compaction_artifact_for_cli(cli)
    assert manual_artifact is not None
    assert manual_artifact["trigger"] == "manual"
    assert manual_artifact["focus_topic"] == "manual focus"


def test_tui_session_compress_uses_plugin_owned_token_status_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules
    stored_states: dict[str, object] = {}
    status_updates: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr(
        sidecar_module,
        "load_operational_checkpoint_cli_config",
        lambda: {
            "emit_compaction_status": True,
            "show_summary_preview": False,
            "summary_preview_chars": 48,
        },
    )
    monkeypatch.setattr(
        sidecar_module,
        "load_compaction_states",
        lambda hermes_home=None: dict(stored_states),
    )

    def save_states(states: dict[str, object], hermes_home: object | None = None) -> None:
        del hermes_home
        stored_states.clear()
        stored_states.update(states)

    monkeypatch.setattr(sidecar_module, "save_compaction_states", save_states)

    class FakeEngine:
        def __init__(self) -> None:
            self.callback: Callable[[dict[str, str | None]], None] | None = None
            self.compression_count = 0
            self.context_length = 400_000
            self.last_checkpoint_summary = ""
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 22_800
            self.last_total_tokens = 22_800
            self.name = "operational_checkpoint"
            self.threshold_tokens = 350_000

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            self.callback = callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            return True

        def persist_usage_snapshot(self) -> None:
            return None

        def compress(
            self,
            messages: list[dict[str, object]],
            current_tokens: int | None = None,
            focus_topic: str | None = None,
        ) -> list[dict[str, object]]:
            del messages, current_tokens
            self.compression_count += 1
            self.last_checkpoint_summary = "Operational state preserved."
            if self.callback is not None:
                self.callback(
                    {
                        "focus_topic": focus_topic,
                        "summary": self.last_checkpoint_summary,
                    }
                )
            return [{"role": "user", "content": "checkpoint"}]

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self._cached_system_prompt = ""
            self._context_pressure_warned_at = 0.0
            self._session_db = None
            self.platform = "tui"
            self.session_id = "session-tui"
            self.tools: list[dict[str, object]] = []

        def _safe_print(self, line: str) -> None:
            raise AssertionError(f"manual TUI compression should not safe-print: {line}")

        def flush_memories(
            self,
            messages: list[dict[str, object]],
            min_turns: int = 0,
        ) -> None:
            del messages, min_turns

        def _invalidate_system_prompt(self) -> None:
            self._cached_system_prompt = ""

        def _build_system_prompt(self, system_message: str) -> str:
            return system_message

        def _compress_context(self, *args: object, **kwargs: object) -> tuple[list[dict[str, object]], str]:
            del args, kwargs
            raise AssertionError("plugin-owned compaction should bypass original _compress_context")

    agent = FakeAgent()
    history = [{"role": "user", "content": "raw"}] * 8
    server_module = ModuleType("tui_gateway.server")
    server_module._sessions = {"sid-tui": {"agent": agent, "history": history}}
    emitted_events: list[tuple[str, str, dict[str, object] | None]] = []

    def status_update(sid: str, kind: str, text: str | None = None) -> None:
        status_updates.append((sid, kind, text))

    def emit(event: str, sid: str, payload: dict[str, object] | None = None) -> None:
        emitted_events.append((event, sid, payload))

    def original_session_compress(rid: object, params: dict[str, object]) -> dict[str, object]:
        del rid
        sid = str(params["session_id"])
        server_module._status_update(
            sid,
            "compressing",
            "⠋ compressing 8 messages (~2,394 tok)…",
        )
        session = server_module._sessions[sid]
        compressed, _system_prompt = session["agent"]._compress_context(
            session["history"],
            "",
            approx_tokens=2_394,
            focus_topic=params.get("focus_topic"),
        )
        session["history"] = compressed
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "removed": 7,
                "before_tokens": 2_394,
                "after_tokens": 999,
                "summary": {
                    "headline": "Compressed: 8 → 1 messages",
                    "token_line": "Rough transcript estimate: ~2,394 → ~999 tokens",
                },
                "messages": compressed,
            },
        }

    server_module._methods = {"session.compress": original_session_compress}
    server_module._emit = emit
    server_module._status_update = status_update
    monkeypatch.setitem(sys.modules, "tui_gateway.server", server_module)
    sidecar_module.install_agent_runtime_bridge(agent_class=FakeAgent)
    sidecar_module._bind_agent_runtime_hooks(agent)
    assert sidecar_module._install_tui_session_compress_response_hook(server_module) is True

    response = server_module._methods["session.compress"](
        1,
        {"session_id": "sid-tui", "focus_topic": "tui focus"},
    )

    expected_before = sidecar_module.estimate_request_tokens_rough(
        history,
        system_prompt="",
        tools=[],
    )
    expected_after = sidecar_module.estimate_request_tokens_rough(
        [{"role": "user", "content": "checkpoint"}],
        system_prompt="",
        tools=[],
    )
    assert status_updates == [
        (
            "sid-tui",
            "status",
            f'Operational Checkpoint: compacting active context budget ~{expected_before:,} tokens, focus: "tui focus"...',
        )
    ]
    assert emitted_events[0] == (
        "tool.start",
        "sid-tui",
        {
            "tool_id": "operational-checkpoint-compress-sid-tui",
            "name": "operational_checkpoint",
            "context": f'active context budget ~{expected_before:,} tokens, focus: "tui focus"',
        },
    )
    assert emitted_events[1][0] == "tool.complete"
    assert emitted_events[1][1] == "sid-tui"
    assert emitted_events[1][2] is not None
    assert emitted_events[1][2]["tool_id"] == "operational-checkpoint-compress-sid-tui"
    assert emitted_events[1][2]["name"] == "operational_checkpoint"
    assert emitted_events[1][2]["summary"] == "active context checkpoint updated"
    assert isinstance(emitted_events[1][2]["duration_s"], float)
    assert "error" not in emitted_events[1][2]
    result = response["result"]
    assert result["before_tokens"] == expected_before
    assert result["after_tokens"] == expected_after
    assert result["summary"]["headline"] == "Operational Checkpoint reduced active context budget"
    assert result["summary"]["token_line"] == (
        f"Active context budget: ~{expected_before:,} → ~{expected_after:,} tokens"
    )
    assert "messages" not in result["summary"]["headline"].lower()
    assert "Rough transcript estimate" not in result["summary"]["token_line"]
    assert "messages" not in result
    artifact = result["operational_checkpoint"]
    assert artifact["trigger"] == "manual"
    assert artifact["tokens_before"] == expected_before
    assert artifact["tokens_after"] == expected_after


def test_runtime_defaults_derives_context_limit_from_runtime_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, helpers_module, _sidecar_module = plugin_modules
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        helpers_module,
        "load_plugin_root_config",
        lambda: {
            "defaults": {
                "model": "summary-model",
                "reasoning_effort": "medium",
                "summary_retry_attempts": 3,
                "compaction_threshold_percent": 0.5,
                "head_preserve_messages": 0,
                "minimum_tail_messages": 0,
                "tail_preserve_tokens": 0,
            }
        },
    )
    monkeypatch.setattr(
        helpers_module,
        "load_config",
        lambda: {
            "model": {
                "default": "runtime-model",
                "provider": "runtime-provider",
                "base_url": "https://provider.example/v1",
            },
            "operational_checkpoint": {},
        },
    )

    def fake_get_model_context_length(**kwargs: object) -> int:
        calls.append(dict(kwargs))
        return 272_000

    monkeypatch.setattr(
        helpers_module,
        "get_model_context_length",
        fake_get_model_context_length,
    )

    defaults = helpers_module.load_runtime_defaults()

    assert defaults["context_limit_tokens"] == 272_000
    assert defaults["compaction_threshold_percent"] == 0.5
    assert defaults["auto_compact_at_tokens"] == 136_000
    assert calls == [
        {
            "model": "runtime-model",
            "base_url": "https://provider.example/v1",
            "api_key": "",
            "config_context_length": None,
            "provider": "runtime-provider",
        }
    ]


def test_hydration_uses_checkpoint_cursor_hash_when_message_count_drifted(
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, helpers_module, sidecar_module = plugin_modules
    raw_history = [
        {"role": "user", "content": "before-0"},
        {"role": "assistant", "content": "before-1"},
        {"role": "user", "content": "cursor"},
        {"role": "assistant", "content": "after-0"},
        {"role": "user", "content": "after-1"},
    ]
    record = helpers_module.PersistedCompactionState(
        checkpoint_id="sid:1:99:cursor",
        compacted_messages=[{"role": "assistant", "content": "checkpoint"}],
        compression_count=1,
        focus_topic=None,
        raw_cursor_message_count=99,
        raw_cursor_message_hash=sidecar_module._message_cursor_hash(raw_history[2]),
        raw_message_count=99,
        summary="summary",
        tokens_after=100,
        tokens_before=1000,
        updated_at=1.0,
    )

    hydrated = sidecar_module._hydrate_messages_from_record(
        raw_messages=raw_history,
        record=record,
    )

    assert hydrated == [
        {"role": "assistant", "content": "checkpoint"},
        {"role": "assistant", "content": "after-0"},
        {"role": "user", "content": "after-1"},
    ]



def test_runtime_defaults_use_repo_threshold_not_stale_user_override(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        helpers_module,
        "load_plugin_root_config",
        lambda: {
            "defaults": {
                "model": "summary-model",
                "reasoning_effort": "medium",
                "summary_retry_attempts": 3,
                "compaction_threshold_percent": 0.85,
                "head_preserve_messages": 0,
                "minimum_tail_messages": 0,
                "tail_preserve_tokens": 0,
            }
        },
    )
    monkeypatch.setattr(
        helpers_module,
        "load_config",
        lambda: {
            "model": {"context_length": 272_000},
            "compression": {"threshold": 0.85},
            "operational_checkpoint": {"compaction_threshold_percent": 0.7},
        },
    )
    monkeypatch.setattr(
        helpers_module,
        "get_model_context_length",
        lambda **kwargs: 272_000,
    )

    defaults = helpers_module.load_runtime_defaults()

    assert defaults["compaction_threshold_percent"] == 0.85
    assert defaults["auto_compact_at_tokens"] == 231_200



def test_update_model_recomputes_threshold_as_percent_of_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 350_000,
            "base_url": "",
            "compaction_threshold_percent": 0.85,
            "config_context_length": 400_000,
            "context_limit_tokens": 400_000,
            "head_preserve_messages": 0,
            "minimum_tail_messages": 0,
            "model": "summary-model",
            "provider": "",
            "reasoning_effort": "medium",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 0,
        },
    )

    engine = compressor_module.OperationalCheckpointCompressor()
    engine.update_model("gpt-5.5", context_length=272_000, provider="openai-codex")

    assert engine.context_length == 272_000
    assert engine.threshold_percent == 0.85
    assert engine.threshold_tokens == 231_200


def test_runtime_defaults_use_plugin_toml_model_and_reasoning_only(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        helpers_module,
        "load_plugin_root_config",
        lambda: {
            "defaults": {
                "model": "plugin-model",
                "reasoning_effort": "high",
                "summary_retry_attempts": 3,
                "context_limit_tokens": 400_000,
                "auto_compact_at_tokens": 350_000,
                "head_preserve_messages": 3,
                "minimum_tail_messages": 3,
                "tail_preserve_tokens": 20_000,
            }
        },
    )
    monkeypatch.setattr(
        helpers_module,
        "load_config",
        lambda: {
            "model": {
                "default": "global-model",
                "provider": "global-provider",
                "base_url": "https://example.invalid",
                "context_length": 999_999,
            },
            "operational_checkpoint": {
                "model": "global-plugin-override",
                "reasoning_effort": "minimal",
            },
        },
    )

    defaults = helpers_module.load_runtime_defaults()

    assert defaults["model"] == "plugin-model"
    assert defaults["reasoning_effort"] == "high"


def test_runtime_defaults_allow_zero_protection_settings(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        helpers_module,
        "load_plugin_root_config",
        lambda: {
            "defaults": {
                "model": "plugin-model",
                "reasoning_effort": "medium",
                "summary_retry_attempts": 3,
                "context_limit_tokens": 400_000,
                "auto_compact_at_tokens": 350_000,
                "head_preserve_messages": 0,
                "minimum_tail_messages": 0,
                "tail_preserve_tokens": 0,
            }
        },
    )
    monkeypatch.setattr(
        helpers_module,
        "load_config",
        lambda: {
            "model": {
                "context_length": 400_000,
            },
            "operational_checkpoint": {},
        },
    )

    defaults = helpers_module.load_runtime_defaults()

    assert defaults["head_preserve_messages"] == 0
    assert defaults["minimum_tail_messages"] == 0
    assert defaults["tail_preserve_tokens"] == 0


def test_runtime_defaults_match_actual_plugin_toml(
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, helpers_module, _sidecar_module = plugin_modules

    defaults = helpers_module.load_runtime_defaults()
    engine = compressor_module.OperationalCheckpointCompressor()

    assert defaults["model"] == "gpt-5.4-mini"
    assert defaults["reasoning_effort"] == "medium"
    assert engine.model == "gpt-5.4-mini"
    assert engine.reasoning_effort == "medium"


def test_update_model_keeps_plugin_summary_model_separate_from_main_runtime(
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    engine = compressor_module.OperationalCheckpointCompressor()
    engine.update_model(
        "gpt-5.5",
        context_length=272_000,
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex/",
        api_key="***",
        api_mode="codex_responses",
    )
    request = engine._build_summary_request(
        [{"role": "user", "content": "summarize me"}],
        focus_topic=None,
    )

    assert engine.model == "gpt-5.4-mini"
    assert request.model == "gpt-5.4-mini"
    assert request.provider == "openai-codex"
    assert request.main_runtime == {
        "api_key": "***",
        "api_mode": "codex_responses",
        "base_url": "https://chatgpt.com/backend-api/codex/",
        "model": "gpt-5.5",
        "provider": "openai-codex",
    }


def test_load_plugin_root_config_falls_back_to_packaged_config(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
    tmp_path: Path,
) -> None:
    _package, _compressor_module, helpers_module, _sidecar_module = plugin_modules

    missing_root_config = tmp_path / "missing-operational-checkpoint.toml"
    packaged_config = tmp_path / "operational_checkpoint.toml"
    packaged_config.write_text(
        "[defaults]\n"
        'model = "packaged-model"\n'
        'reasoning_effort = "low"\n'
        "summary_retry_attempts = 3\n"
        "compaction_threshold_percent = 0.85\n"
        "head_preserve_messages = 3\n"
        "minimum_tail_messages = 3\n"
        "tail_preserve_tokens = 20000\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(helpers_module, "PLUGIN_CONFIG_PATH", missing_root_config)
    monkeypatch.setattr(
        helpers_module,
        "PACKAGED_PLUGIN_CONFIG_PATH",
        packaged_config,
        raising=False,
    )

    config = helpers_module.load_plugin_root_config()

    assert config["defaults"]["model"] == "packaged-model"
    assert config["defaults"]["reasoning_effort"] == "low"


def test_pyproject_exposes_hermes_entrypoint_and_packaged_config() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    entrypoints = pyproject["project"]["entry-points"]["hermes_agent.plugins"]
    assert entrypoints["operational_checkpoint"] == "operational_checkpoint"

    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert "operational_checkpoint.toml" in package_data["operational_checkpoint"]


def test_helpers_call_llm_forwards_main_runtime(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, helpers_module, _sidecar_module = plugin_modules
    auxiliary_module = importlib.import_module("agent.auxiliary_client")
    captured: dict[str, object] = {}

    def fake_call_llm(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(choices=[])

    monkeypatch.setattr(auxiliary_module, "call_llm", fake_call_llm)

    main_runtime = {
        "model": "runtime-model",
        "provider": "custom",
        "base_url": "http://127.0.0.1:8765/v1",
        "api_key": "dummy-key",
        "api_mode": "chat_completions",
    }
    helpers_module.call_llm(
        task="compression",
        provider="custom",
        model="runtime-model",
        base_url="http://127.0.0.1:8765/v1",
        api_key="dummy-key",
        messages=[{"role": "user", "content": "checkpoint me"}],
        max_tokens=None,
        extra_body={"reasoning": {"effort": "medium"}},
        main_runtime=main_runtime,
    )

    assert captured["provider"] == "custom"
    assert captured["model"] == "runtime-model"
    assert captured["base_url"] == "http://127.0.0.1:8765/v1"
    assert captured["api_key"] == "dummy-key"
    assert captured["main_runtime"] == main_runtime
    assert captured["task"] == "compression"


def test_generate_summary_passes_current_runtime_to_auxiliary_llm(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "medium",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    captured: dict[str, object] = {}

    def fake_call_llm(
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
        del messages, max_tokens, extra_body
        captured["task"] = task
        captured["provider"] = provider
        captured["model"] = model
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["main_runtime"] = main_runtime
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=checkpoint_summary("Continue the task")
                    )
                )
            ]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    engine.update_model(
        "runtime-model",
        context_length=200_000,
        base_url="http://127.0.0.1:8765/v1",
        api_key="dummy-key",
        provider="custom",
        api_mode="chat_completions",
    )
    result = engine._generate_summary(
        [{"role": "user", "content": "Investigate the adapter failure"}],
        focus_topic="adapter failure",
    )

    assert captured["task"] == "compression"
    assert captured["provider"] == "custom"
    assert captured["model"] == "gpt-5.4"
    assert captured["base_url"] == "http://127.0.0.1:8765/v1"
    assert captured["api_key"] == "dummy-key"
    assert captured["main_runtime"] == {
        "api_key": "dummy-key",
        "api_mode": "chat_completions",
        "base_url": "http://127.0.0.1:8765/v1",
        "model": "runtime-model",
        "provider": "custom",
    }
    assert result is not None


def test_generate_summary_infers_custom_provider_from_base_url(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "http://127.0.0.1:8765/v1",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "medium",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    captured: dict[str, object] = {}

    def fake_call_llm(
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
        del task, messages, max_tokens, extra_body
        captured["provider"] = provider
        captured["model"] = model
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["main_runtime"] = main_runtime
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=checkpoint_summary("Continue")))]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    result = engine._generate_summary(
        [{"role": "user", "content": "Investigate the adapter failure"}],
        focus_topic=None,
    )

    assert captured["provider"] == "custom"
    assert captured["model"] == "gpt-5.4"
    assert captured["base_url"] == "http://127.0.0.1:8765/v1"
    assert captured["api_key"] is None
    assert captured["main_runtime"] == {
        "api_key": "",
        "api_mode": "",
        "base_url": "http://127.0.0.1:8765/v1",
        "model": "gpt-5.4",
        "provider": "custom",
    }
    assert result is not None


def test_generate_summary_explicit_runtime_overrides_upstream_auxiliary_compression_defaults(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "https://chatgpt.com/backend-api/codex",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4-mini",
            "provider": "openai-codex",
            "reasoning_effort": "medium",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    captured: dict[str, object] = {}

    def fake_call_llm(
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
        del messages, max_tokens, extra_body
        captured["task"] = task
        captured["provider"] = provider
        captured["model"] = model
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["main_runtime"] = main_runtime
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=checkpoint_summary("Continue")))]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    result = engine._generate_summary(
        [{"role": "user", "content": "Investigate the adapter failure"}],
        focus_topic=None,
    )

    assert captured["task"] == "compression"
    assert captured["provider"] == "openai-codex"
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["base_url"] is None
    assert captured["api_key"] is None
    assert captured["main_runtime"] == {
        "api_key": "",
        "api_mode": "",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "model": "gpt-5.4-mini",
        "provider": "openai-codex",
    }
    assert result is not None


def test_generate_summary_does_not_force_codex_base_url_as_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "https://chatgpt.com/backend-api/codex",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4-mini",
            "provider": "openai-codex",
            "reasoning_effort": "medium",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    captured: dict[str, object] = {}

    def fake_call_llm(
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
        del task, messages, max_tokens, extra_body
        captured["provider"] = provider
        captured["model"] = model
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["main_runtime"] = main_runtime
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=checkpoint_summary("Continue")))]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    result = engine._generate_summary(
        [{"role": "user", "content": "Investigate the adapter failure"}],
        focus_topic=None,
    )

    assert captured["provider"] == "openai-codex"
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["base_url"] is None
    assert captured["api_key"] is None
    assert captured["main_runtime"] == {
        "api_key": "",
        "api_mode": "",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "model": "gpt-5.4-mini",
        "provider": "openai-codex",
    }
    assert result is not None


def test_compressor_restores_persisted_usage_and_rolls_over_child_session(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
    tmp_path: Path,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    engine = compressor_module.OperationalCheckpointCompressor()
    engine.on_session_start("session-parent", hermes_home=str(tmp_path))
    engine.update_from_response(
        {
            "prompt_tokens": 240,
            "completion_tokens": 30,
            "total_tokens": 270,
        }
    )

    restored = compressor_module.OperationalCheckpointCompressor()
    restored.on_session_start("session-parent", hermes_home=str(tmp_path))

    assert restored.last_prompt_tokens == 240
    assert restored.last_completion_tokens == 30
    assert restored.last_total_tokens == 270

    restored.rollover_usage_snapshot(
        new_session_id="session-child",
        post_compaction_tokens=91,
    )

    child = compressor_module.OperationalCheckpointCompressor()
    child.on_session_start("session-child", hermes_home=str(tmp_path))

    assert child.last_prompt_tokens == 91
    assert child.last_completion_tokens == 0
    assert child.last_total_tokens == 91


def test_compressor_should_compress_prefers_current_tokens_when_provided(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    engine = compressor_module.OperationalCheckpointCompressor()
    engine.last_prompt_tokens = 90_000
    engine.last_completion_tokens = 10_000
    engine.last_total_tokens = 100_000

    assert engine.should_compress(prompt_tokens=999_999) is True

    engine.last_total_tokens = 200_000

    assert engine.should_compress(prompt_tokens=0) is False


def test_compressor_snapshot_payload_writes_only_behavioral_fields(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 175_000,
            "base_url": "",
            "config_context_length": 200_000,
            "context_limit_tokens": 200_000,
            "head_preserve_messages": 2,
            "minimum_tail_messages": 3,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 18_000,
        },
    )

    engine = compressor_module.OperationalCheckpointCompressor()
    engine.compression_count = 2
    engine.last_checkpoint_focus_topic = "focus topic"
    engine.last_checkpoint_summary = "checkpoint body"
    engine.last_prompt_tokens = 240
    engine.last_completion_tokens = 30
    engine.last_total_tokens = 270

    payload = engine._snapshot_payload()

    assert set(payload) == {
        "compression_count",
        "last_checkpoint_focus_topic",
        "last_checkpoint_summary",
        "last_completion_tokens",
        "last_prompt_tokens",
        "last_total_tokens",
        "updated_at",
    }


def test_compress_allows_zero_head_and_tail_and_summarizes_full_window(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 900,
            "base_url": "",
            "config_context_length": 1_000,
            "context_limit_tokens": 1_000,
            "head_preserve_messages": 0,
            "minimum_tail_messages": 0,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 0,
        },
    )

    captured: dict[str, object] = {}

    def fake_call_llm(
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
        del provider, model, base_url, api_key, task, max_tokens, extra_body, main_runtime
        captured["prompt"] = messages[0]["content"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=checkpoint_summary("compact the whole working view")
                    )
                )
            ]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    history = [
        {"role": "user", "content": "first-user payload"},
        {"role": "assistant", "content": "first-assistant payload"},
        {"role": "user", "content": "last-user payload"},
        {"role": "assistant", "content": "last-assistant payload"},
    ]

    compacted = engine.compress(history)

    assert engine.protect_first_n == 0
    assert engine.protect_last_n == 0
    assert engine.tail_token_budget == 0
    assert len(compacted) == 1
    assert compacted[0]["role"] == "assistant"
    assert str(compacted[0]["content"]).startswith(compressor_module.SUMMARY_PREFIX)
    prompt = str(captured["prompt"])
    assert "first-user payload" in prompt
    assert "last-assistant payload" in prompt


def test_zero_minimum_tail_still_respects_positive_tail_token_budget(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, compressor_module, _helpers_module, _sidecar_module = plugin_modules

    monkeypatch.setattr(
        compressor_module,
        "load_runtime_defaults",
        lambda: {
            "auto_compact_at_tokens": 900,
            "base_url": "",
            "config_context_length": 1_000,
            "context_limit_tokens": 1_000,
            "head_preserve_messages": 1,
            "minimum_tail_messages": 0,
            "model": "gpt-5.4",
            "provider": "",
            "reasoning_effort": "",
            "summary_retry_attempts": 3,
            "tail_preserve_tokens": 1_000,
        },
    )

    captured: dict[str, object] = {}

    def fake_call_llm(
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
        del provider, model, base_url, api_key, task, max_tokens, extra_body, main_runtime
        captured["prompt"] = messages[0]["content"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=checkpoint_summary("keep a raw tail")
                    )
                )
            ]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_call_llm)

    engine = compressor_module.OperationalCheckpointCompressor()
    history = [
        {"role": "user", "content": "head-anchor"},
        {"role": "assistant", "content": "middle-one"},
        {"role": "user", "content": "middle-two"},
        {"role": "assistant", "content": "tail-three"},
        {"role": "user", "content": "tail-four"},
    ]

    compacted = engine.compress(history)

    assert compacted[0]["content"] == "head-anchor"
    assert compacted[-1]["content"] == "tail-four"
    assert any(
        str(message.get("content", "")).startswith(compressor_module.SUMMARY_PREFIX)
        for message in compacted
    )
    prompt = str(captured["prompt"])
    assert "middle-one" in prompt
    assert "tail-four" not in prompt


def test_sidecar_init_does_not_seed_missing_snapshot_from_loaded_history(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules

    monkeypatch.setattr(
        sidecar_module,
        "load_operational_checkpoint_cli_config",
        lambda: {
            "emit_compaction_status": True,
            "show_summary_preview": False,
            "summary_preview_chars": 48,
        },
    )
    monkeypatch.setattr(
        sidecar_module,
        "load_compaction_states",
        lambda hermes_home=None: {},
    )
    monkeypatch.setattr(sidecar_module, "save_compaction_states", lambda *args, **kwargs: None)

    class FakeEngine:
        def __init__(self) -> None:
            self.callback: Callable[[dict[str, str | None]], None] | None = None
            self.compression_count = 0
            self.context_length = 400_000
            self.last_checkpoint_summary = ""
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.restore_calls = 0
            self.threshold_tokens = 350_000

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            self.callback = callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            self.restore_calls += 1
            return False

        def persist_usage_snapshot(self) -> None:
            return None

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self.session_id = "session-seed"
            self.conversation_history = [
                {"role": "user", "content": "alpha " + ("x" * 160)},
                {"role": "assistant", "content": "beta " + ("y" * 120)},
            ]

        def _compress_context(
            self,
            messages: list[dict[str, object]],
            system_message: str,
            **runtime_kwargs: object,
        ) -> tuple[list[dict[str, object]], str]:
            del messages, runtime_kwargs
            return ([{"role": "user", "content": "checkpoint"}], system_message)

    class FakeCLI:
        def __init__(self) -> None:
            self.agent = FakeAgent()

        def _init_agent(self, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            return True

        def _manual_compress(self, cmd_original: str = "") -> None:
            del cmd_original

    sidecar_module.install_runtime_bridge(cli_class=FakeCLI, agent_class=FakeAgent)

    cli = FakeCLI()
    assert cli._init_agent() is True
    assert cli.agent.context_compressor.restore_calls == 1
    assert cli.agent.context_compressor.last_prompt_tokens == 0
    assert cli.agent.context_compressor.last_completion_tokens == 0


def test_sidecar_init_binds_engine_to_upstream_selected_child_session(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules

    monkeypatch.setattr(
        sidecar_module,
        "load_operational_checkpoint_cli_config",
        lambda: {
            "emit_compaction_status": True,
            "show_summary_preview": False,
            "summary_preview_chars": 48,
        },
    )
    monkeypatch.setattr(
        sidecar_module,
        "load_compaction_states",
        lambda hermes_home=None: {},
    )
    monkeypatch.setattr(sidecar_module, "save_compaction_states", lambda *args, **kwargs: None)

    class FakeEngine:
        def __init__(self) -> None:
            self.bound_session_id = ""
            self.callback: Callable[[dict[str, str | None]], None] | None = None
            self.compression_count = 0
            self.last_checkpoint_summary = ""
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.restore_calls = 0
            self.threshold_tokens = 350_000

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            self.callback = callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            self.restore_calls += 1
            return True

        def persist_usage_snapshot(self) -> None:
            return None

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self.printed: list[str] = []
            self.session_id = "session-root"

        def _safe_print(self, line: str) -> None:
            self.printed.append(line)

        def _compress_context(
            self,
            messages: list[dict[str, object]],
            system_message: str,
            **runtime_kwargs: object,
        ) -> tuple[list[dict[str, object]], str]:
            del messages, runtime_kwargs
            return ([{"role": "user", "content": "checkpoint"}], system_message)

    class FakeCLI:
        def __init__(self) -> None:
            self.agent = FakeAgent()
            self.session_id = "session-root"

        def _init_agent(self, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            self.session_id = "session-child"
            self.agent.session_id = "session-child"
            return True

        def _manual_compress(self, cmd_original: str = "") -> None:
            del cmd_original

    sidecar_module.install_runtime_bridge(cli_class=FakeCLI, agent_class=FakeAgent)

    cli = FakeCLI()
    assert cli._init_agent() is True
    assert cli.session_id == "session-child"
    assert cli.agent.session_id == "session-child"
    assert cli.agent.context_compressor.bound_session_id == "session-child"
    assert cli.agent.context_compressor.restore_calls == 1
    assert cli.agent.printed == []


def test_sidecar_init_binds_engine_to_upstream_selected_lineage_leaf(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules

    monkeypatch.setattr(
        sidecar_module,
        "load_operational_checkpoint_cli_config",
        lambda: {
            "emit_compaction_status": True,
            "show_summary_preview": False,
            "summary_preview_chars": 48,
        },
    )
    monkeypatch.setattr(
        sidecar_module,
        "load_compaction_states",
        lambda hermes_home=None: {},
    )
    monkeypatch.setattr(sidecar_module, "save_compaction_states", lambda *args, **kwargs: None)

    class FakeEngine:
        def __init__(self) -> None:
            self.bound_session_id = ""
            self.callback: Callable[[dict[str, str | None]], None] | None = None
            self.compression_count = 0
            self.last_checkpoint_summary = ""
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.threshold_tokens = 350_000

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            self.callback = callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            return True

        def persist_usage_snapshot(self) -> None:
            return None

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self.printed: list[str] = []
            self.session_id = "session-root"

        def _safe_print(self, line: str) -> None:
            self.printed.append(line)

        def _compress_context(
            self,
            messages: list[dict[str, object]],
            system_message: str,
            **runtime_kwargs: object,
        ) -> tuple[list[dict[str, object]], str]:
            del messages, runtime_kwargs
            return ([{"role": "user", "content": "checkpoint"}], system_message)

    class FakeCLI:
        def __init__(self) -> None:
            self.agent = FakeAgent()
            self.session_id = "session-root"

        def _init_agent(self, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            self.session_id = "session-grandchild"
            self.agent.session_id = "session-grandchild"
            return True

        def _manual_compress(self, cmd_original: str = "") -> None:
            del cmd_original

    sidecar_module.install_runtime_bridge(cli_class=FakeCLI, agent_class=FakeAgent)

    cli = FakeCLI()
    assert cli._init_agent() is True
    assert cli.session_id == "session-grandchild"
    assert cli.agent.session_id == "session-grandchild"
    assert cli.agent.context_compressor.bound_session_id == "session-grandchild"
    assert cli.agent.printed == []


def test_sidecar_manual_compress_keeps_same_session_and_preserves_raw_upstream_transcript(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules
    stored_states: dict[str, object] = {}

    monkeypatch.setattr(
        sidecar_module,
        "load_operational_checkpoint_cli_config",
        lambda: {
            "emit_compaction_status": True,
            "show_summary_preview": False,
            "summary_preview_chars": 48,
        },
    )
    monkeypatch.setattr(
        sidecar_module,
        "load_compaction_states",
        lambda hermes_home=None: dict(stored_states),
    )

    def save_states(states: dict[str, object], hermes_home: object | None = None) -> None:
        del hermes_home
        stored_states.clear()
        stored_states.update(states)

    monkeypatch.setattr(sidecar_module, "save_compaction_states", save_states)

    class FakeEngine:
        def __init__(self) -> None:
            self.callback: Callable[[dict[str, str | None]], None] | None = None
            self.compression_count = 0
            self.context_length = 400_000
            self.last_checkpoint_summary = ""
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.persist_calls = 0
            self.threshold_tokens = 350_000

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            self.callback = callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            return True

        def persist_usage_snapshot(self) -> None:
            self.persist_calls += 1

        def compress(
            self,
            messages: list[dict[str, object]],
            current_tokens: int | None = None,
            focus_topic: str | None = None,
        ) -> list[dict[str, object]]:
            del messages, current_tokens
            self.compression_count += 1
            self.last_checkpoint_summary = checkpoint_summary("stay on this session")
            if self.callback is not None:
                self.callback(
                    {
                        "focus_topic": focus_topic,
                        "summary": self.last_checkpoint_summary,
                    }
                )
            return [{"role": "user", "content": "checkpoint"}]

    class FakeSessionDB:
        def __init__(self) -> None:
            self.appended_messages: list[dict[str, object]] = []
            self.ensure_calls: list[tuple[str, str, str]] = []
            self.raw_messages: list[dict[str, object]] = [
                {"role": "user", "content": "manual"}
            ]

        def ensure_session(self, session_id: str, source: str, model: str) -> None:
            self.ensure_calls.append((session_id, source, model))

        def append_message(self, **kwargs: object) -> None:
            self.appended_messages.append(
                {
                    "role": kwargs.get("role"),
                    "content": kwargs.get("content"),
                }
            )
            self.raw_messages.append(
                {
                    "role": kwargs.get("role"),
                    "content": kwargs.get("content"),
                }
            )

        def get_messages_as_conversation(self, session_id: str) -> list[dict[str, object]]:
            del session_id
            return list(self.raw_messages)

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self._cached_system_prompt = ""
            self._context_pressure_warned_at = 0.0
            self._last_flushed_db_idx = 0
            self._session_db = FakeSessionDB()
            self.flushed_messages: list[dict[str, object]] = []
            self.session_id = "session-parent"
            self.tools: list[dict[str, object]] = []

        def flush_memories(
            self,
            messages: list[dict[str, object]],
            min_turns: int = 0,
        ) -> None:
            del messages, min_turns

        def _invalidate_system_prompt(self) -> None:
            self._cached_system_prompt = ""

        def _build_system_prompt(self, system_message: str) -> str:
            return f"{system_message}::built"

        def _save_session_log(self, messages: list[dict[str, object]]) -> None:
            self.saved_messages = list(messages)

        def _flush_messages_to_session_db(
            self,
            messages: list[dict[str, object]],
            conversation_history: list[dict[str, object]] | None = None,
        ) -> None:
            del conversation_history
            self.flushed_messages = list(messages)
            self._last_flushed_db_idx = len(messages)

        def _compress_context(self, *args: object, **kwargs: object) -> tuple[list[dict[str, object]], str]:
            del args, kwargs
            raise AssertionError("plugin-owned compaction should bypass the original _compress_context")

    class FakeCLI:
        def __init__(self) -> None:
            self.agent = FakeAgent()
            self.session_id = "session-parent"

        def _init_agent(self, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            return True

        def _manual_compress(self, cmd_original: str = "") -> None:
            del cmd_original
            self.agent._compress_context(
                [{"role": "user", "content": "manual"}],
                "",
                approx_tokens=123_000,
                focus_topic="manual focus",
            )

    sidecar_module.install_runtime_bridge(cli_class=FakeCLI, agent_class=FakeAgent)

    cli = FakeCLI()
    assert cli._init_agent() is True

    cli._manual_compress()
    expected_tokens = sidecar_module.estimate_request_tokens_rough(
        [{"role": "user", "content": "checkpoint"}],
        system_prompt="::built",
        tools=[],
    )

    assert cli.session_id == "session-parent"
    assert cli.agent.session_id == "session-parent"
    assert cli.agent.context_compressor.bound_session_id == "session-parent"
    assert cli.agent.context_compressor.persist_calls == 1
    assert cli.agent._session_db.appended_messages == []
    stored_record = stored_states["session-parent"]
    assert getattr(stored_record, "raw_message_count") == 1
    assert getattr(stored_record, "compacted_messages") == [
        {"role": "user", "content": "checkpoint"}
    ]

    cli.agent._save_session_log(
        [
            {"role": "user", "content": "checkpoint"},
            {"role": "assistant", "content": "continued"},
        ]
    )
    cli.agent._flush_messages_to_session_db(
        [
            {"role": "user", "content": "checkpoint"},
            {"role": "assistant", "content": "continued"},
        ],
        conversation_history=[{"role": "user", "content": "manual"}],
    )

    assert cli.agent.saved_messages == [
        {"role": "user", "content": "manual"},
        {"role": "assistant", "content": "continued"},
    ]
    assert cli.agent._session_db.appended_messages == [
        {"role": "assistant", "content": "continued"}
    ]
    assert cli.agent.context_compressor.last_prompt_tokens == expected_tokens
    assert cli.agent.context_compressor.last_total_tokens == expected_tokens


def test_sidecar_init_hydrates_resumed_history_from_plugin_state(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
) -> None:
    _package, _compressor_module, _helpers_module, sidecar_module = plugin_modules

    monkeypatch.setattr(
        sidecar_module,
        "load_operational_checkpoint_cli_config",
        lambda: {
            "emit_compaction_status": True,
            "show_summary_preview": False,
            "summary_preview_chars": 48,
        },
    )

    stored_record = sidecar_module.PersistedCompactionState(
        compacted_messages=[{"role": "assistant", "content": "checkpoint"}],
        compression_count=2,
        focus_topic="resume",
        raw_message_count=4,
        summary="checkpoint",
        tokens_after=11,
        tokens_before=99,
        updated_at=1.0,
    )
    monkeypatch.setattr(
        sidecar_module,
        "load_compaction_states",
        lambda hermes_home=None: {"session-child": stored_record},
    )
    monkeypatch.setattr(sidecar_module, "save_compaction_states", lambda *args, **kwargs: None)

    class FakeEngine:
        def __init__(self) -> None:
            self.bound_session_id = ""
            self.callback: Callable[[dict[str, str | None]], None] | None = None
            self.compression_count = 0
            self.last_checkpoint_summary = ""
            self.last_completion_tokens = 0
            self.last_prompt_tokens = 0
            self.last_total_tokens = 0
            self.name = "operational_checkpoint"
            self.threshold_tokens = 350_000

        def set_compaction_callback(
            self,
            callback: Callable[[dict[str, str | None]], None] | None,
        ) -> None:
            self.callback = callback

        def bind_session(
            self,
            *,
            session_id: str,
            hermes_home: str | None = None,
            parent_session_id: str | None = None,
        ) -> None:
            del hermes_home, parent_session_id
            self.bound_session_id = session_id

        def restore_usage_snapshot(self) -> bool:
            return True

        def persist_usage_snapshot(self) -> None:
            return None

    raw_history = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    class FakeAgent:
        def __init__(self) -> None:
            self.context_compressor = FakeEngine()
            self.session_id = "session-root"
            self.tools: list[dict[str, object]] = []

        def _compress_context(
            self,
            messages: list[dict[str, object]],
            system_message: str,
            **runtime_kwargs: object,
        ) -> tuple[list[dict[str, object]], str]:
            del messages, runtime_kwargs
            return ([{"role": "user", "content": "checkpoint"}], system_message)

    class FakeCLI:
        def __init__(self) -> None:
            self.agent = FakeAgent()
            self.conversation_history = list(raw_history)
            self.session_id = "session-root"

        def _init_agent(self, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            self.session_id = "session-child"
            self.agent.session_id = "session-child"
            return True

        def _manual_compress(self, cmd_original: str = "") -> None:
            del cmd_original

        def _handle_resume_command(self, cmd_original: str) -> None:
            del cmd_original

    sidecar_module.install_runtime_bridge(cli_class=FakeCLI, agent_class=FakeAgent)

    cli = FakeCLI()
    assert cli._init_agent() is True

    expected_history = [
        {"role": "assistant", "content": "checkpoint"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    expected_tokens = sidecar_module.estimate_request_tokens_rough(
        expected_history,
        system_prompt="",
        tools=[],
    )

    assert cli.conversation_history == expected_history
    assert cli.agent.context_compressor.bound_session_id == "session-child"
    assert cli.agent.context_compressor.last_prompt_tokens == expected_tokens
    assert sidecar_module.hydrated_cursor_for_agent(cli.agent) == 1


def test_auto_preflight_compaction_continues_turn(
    monkeypatch: pytest.MonkeyPatch,
    plugin_modules: PluginModules,
    tmp_path: Path,
) -> None:
    _package, compressor_module, helpers_module, _sidecar_module = plugin_modules
    import hermes_cli.plugins as plugins_mod
    import run_agent
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  context_length: 70000\n"
        "compression:\n"
        "  enabled: true\n"
        "context:\n"
        "  engine: operational_checkpoint\n"
        "operational_checkpoint:\n"
        "  compaction_threshold_percent: 0.005\n"
        "  head_preserve_messages: 3\n"
        "  minimum_tail_messages: 3\n"
        "  tail_preserve_tokens: 200\n"
        "  summary_retry_attempts: 3\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})
    monkeypatch.setattr(
        helpers_module,
        "load_plugin_root_config",
        lambda: {
            "defaults": {
                "model": "gpt-5.4-mini",
                "reasoning_effort": "medium",
                "summary_retry_attempts": 3,
                "compaction_threshold_percent": 0.005,
                "head_preserve_messages": 3,
                "minimum_tail_messages": 3,
                "tail_preserve_tokens": 200,
            }
        },
    )

    def fake_summary_call_llm(
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
        del provider, model, base_url, api_key, task, messages, max_tokens, extra_body, main_runtime
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=checkpoint_summary("Preserve operational state")
                    )
                )
            ]
        )

    monkeypatch.setattr(compressor_module, "call_llm", fake_summary_call_llm)

    old_manager = plugins_mod._plugin_manager
    manager = PluginManager()
    plugins_mod._plugin_manager = manager
    try:
        ctx = PluginContext(PluginManifest(name="operational_checkpoint"), manager)
        ctx.register_context_engine(compressor_module.OperationalCheckpointCompressor())

        class TestAgent(run_agent.AIAgent):
            def __init__(self, *args: object, **kwargs: object) -> None:
                kwargs.update(
                    skip_context_files=True,
                    skip_memory=True,
                    max_iterations=4,
                    enabled_toolsets=[],
                    quiet_mode=True,
                )
                super().__init__(*args, **kwargs)
                self._cleanup_task_resources = lambda *a, **k: None
                self._persist_session = lambda *a, **k: None
                self._save_trajectory = lambda *a, **k: None
                self._save_session_log = lambda *a, **k: None

            def run_conversation(
                self,
                msg: str,
                conversation_history: list[dict[str, object]] | None = None,
                task_id: str | None = None,
            ) -> dict[str, object]:
                self._interruptible_api_call = lambda kw: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            index=0,
                            message=SimpleNamespace(
                                role="assistant",
                                content="continued after compaction",
                                tool_calls=None,
                                reasoning_content=None,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=120,
                        completion_tokens=20,
                        total_tokens=140,
                    ),
                    model="test-model",
                )
                self._disable_streaming = True
                return super().run_conversation(
                    msg,
                    conversation_history=conversation_history,
                    task_id=task_id,
                )

        history: list[dict[str, object]] = []
        big_chunk = "x" * 900
        for idx in range(4):
            history.append({"role": "user", "content": f"user-{idx} {big_chunk}"})
            history.append({"role": "assistant", "content": f"assistant-{idx} {big_chunk}"})

        agent = TestAgent(
            model="test-model",
            base_url="http://127.0.0.1:8765/v1",
            api_key="dummy-key",
            provider="custom",
            api_mode="chat_completions",
            platform="cli",
        )
        original_session_id = agent.session_id
        result = agent.run_conversation("final turn", conversation_history=history)

        assert agent.context_compressor.name == "operational_checkpoint"
        assert agent.context_compressor.compression_count == 1
        assert agent.context_compressor.last_checkpoint_summary.startswith("1. Objective")
        assert agent.session_id == original_session_id
        assert result["final_response"] == "continued after compaction"
    finally:
        plugins_mod._plugin_manager = old_manager
