from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

PLUGIN_NAME: str = "operational_checkpoint"


def _default_config_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "config.yaml"
    except Exception:
        return Path.home() / ".hermes" / "config.yaml"


def _read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _enabled_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    enabled: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip() and item not in enabled:
            enabled.append(item)
    return enabled


def ensure_plugin_activation(config_path: str | Path | None = None) -> bool:
    """Enable this plugin in Hermes config using real YAML structures.

    Returns True when the file changed. This intentionally does not mutate
    model.context_length, compression thresholds, or operational_checkpoint
    budget defaults; it only activates the plugin and selects the context engine.
    """
    path = Path(config_path).expanduser() if config_path is not None else _default_config_path()
    config = _read_config(path)
    original = yaml.safe_dump(config, sort_keys=False)

    plugins = config.get("plugins")
    if not isinstance(plugins, dict):
        plugins = {}
        config["plugins"] = plugins

    enabled = _enabled_list(plugins.get("enabled"))
    if PLUGIN_NAME not in enabled:
        enabled.append(PLUGIN_NAME)
    plugins["enabled"] = enabled

    context = config.get("context")
    if not isinstance(context, dict):
        context = {}
        config["context"] = context
    context["engine"] = PLUGIN_NAME

    updated = yaml.safe_dump(config, sort_keys=False)
    if updated == original:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enable the Hermes Operational Checkpoint plugin in config.yaml."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to Hermes config.yaml. Defaults to $HERMES_HOME/config.yaml or ~/.hermes/config.yaml.",
    )
    args = parser.parse_args(argv)

    config_path = args.config if args.config is not None else _default_config_path()
    changed = ensure_plugin_activation(config_path)
    action = "Updated" if changed else "Already configured"
    print(f"{action}: {config_path}")
    print(f"- plugins.enabled includes {PLUGIN_NAME}")
    print(f"- context.engine = {PLUGIN_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
