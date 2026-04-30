# Hermes Operational Checkpoint Plugin

![Hermes Operational Checkpoint Plugin banner](./Operational-Checkpoint.jpg)

Operational Checkpoint keeps long Hermes sessions usable after compression.

Normal summaries are fine when you only need the gist. They are not fine when a session is carrying real work: decisions, constraints, failed paths, active files, evidence, risks, and the next move. Lose those, and the model does not continue the work. It starts reconstructing it.

This plugin turns compression into an operational checkpoint. The goal is not a nicer recap. The goal is that the next turn still knows what is going on.

Use it for long debugging runs, codebase surgery, research threads, multi-step implementation, and agent work that may compact more than once.

What you get:

- dense continuation state instead of a soft summary
- same-session compression instead of a child-session handoff
- checkpoint validation, so malformed summaries do not quietly become memory
- safe fallback checkpoints when summary generation fails
- CLI/TUI hooks for manual compression, auto-compaction, resume, save, hydration, and usage display
- a helper command that activates the plugin with real YAML, not a config value that only looks right

## Why this became more than one hook

Hermes gives plugins a context-engine hook. That is the right doorway, but it is not the whole room.

A hook can register a compressor. It cannot, by itself, make sure Hermes actually loads the plugin, that `/compress` and auto-compaction hit the same path, that the work stays in the same session, that resumed sessions hydrate the checkpoint correctly, or that the TUI stops showing stale usage after compaction.

So this repo reaches into the real runtime boundary: activation, sidecar hooks, CLI/TUI paths, resume, save, hydration, checkpoint validation, and fallback shape. Without that, Operational Checkpoint would be a nice engine class that only works when Hermes enters through the happy path.

That is not the goal. The goal is continuity that survives real sessions. Compression should make the active context smaller without sanding off the operational state of the work.

## How it works

When `context.engine: operational_checkpoint` is active, this plugin becomes the compression engine for the session.

It writes an 11-section checkpoint focused on continuation:

1. Objective
2. Explicit user instructions / prohibitions / scope boundaries
3. Operational state
4. Active working set
5. Discoveries / evidence
6. Settled decisions / rejected alternatives
7. Transferable patterns learned this run
8. Assumptions / uncertainties / blockers
9. Execution status
10. Action frontier
11. Critical invariants / regression risks

The checkpoint uses explicit epistemic labels:

- `[Observed]`
- `[Inferred]`
- `[Assumption]`
- `[Unknown]`
- `[Blocked]`

That structure is deliberate. It keeps the checkpoint from turning into prose. A continuation model needs state, evidence, boundaries, and next action. It does not need a bedtime story about the session.

Operational Checkpoint also owns its checkpoint-generation runtime, so compression is not quietly steered by upstream auxiliary compression defaults. Changing the main chat model should not accidentally change how checkpointing behaves.

By default, the plugin checkpoints the full compression window. It does not preserve raw head or tail anchors unless you opt into that. Raw history is still retained separately; the active working view gets tighter.

## Install

Install from GitHub into the Hermes virtualenv and activate it:

```bash
uv pip install \
  --python ~/.hermes/hermes-agent/venv/bin/python \
  "git+https://github.com/ReyJ94/Hermes-Operational-Checkpoint-Plugin.git" && \
  ~/.hermes/hermes-agent/venv/bin/operational-checkpoint-config
```

If you do not use `uv`:

```bash
~/.hermes/hermes-agent/venv/bin/pip install \
  "git+https://github.com/ReyJ94/Hermes-Operational-Checkpoint-Plugin.git" && \
  ~/.hermes/hermes-agent/venv/bin/operational-checkpoint-config
```

The helper writes the activation state Hermes actually needs:

```yaml
plugins:
  enabled:
    - operational_checkpoint
context:
  engine: operational_checkpoint
```

Do not activate the plugin with this:

```bash
hermes config set plugins.enabled '["operational_checkpoint"]'
```

On current Hermes builds that can write a string instead of a YAML list. The config then looks plausible, but plugin loading still skips the entry-point plugin. The helper exists because that bug is easy to miss and annoying to diagnose.

For a non-default config path:

```bash
~/.hermes/hermes-agent/venv/bin/operational-checkpoint-config \
  --config ~/.hermes/profiles/<profile-name>/config.yaml
```

The helper only activates the plugin. It does not change model context length, thresholds, providers, or compression-budget defaults.

## Local development install

From a local checkout:

```bash
cd /path/to/operational-checkpoint
uv build --wheel --out-dir dist
uv pip install \
  --python ~/.hermes/hermes-agent/venv/bin/python \
  --force-reinstall \
  dist/operational_checkpoint-*.whl && \
  ~/.hermes/hermes-agent/venv/bin/operational-checkpoint-config
```

Plain `pip` works too:

```bash
cd /path/to/operational-checkpoint
python -m pip install build
python -m build --wheel --out-dir dist
~/.hermes/hermes-agent/venv/bin/pip install \
  --force-reinstall \
  dist/operational_checkpoint-*.whl && \
  ~/.hermes/hermes-agent/venv/bin/operational-checkpoint-config
```

After local edits, rebuild and reinstall before testing against Hermes. Hermes will not pick up source changes from this checkout until the installed wheel is refreshed.

## Defaults

The shipped defaults live in `operational_checkpoint.toml`:

```toml
[defaults]
model = "gpt-5.4-mini"
reasoning_effort = "medium"
summary_retry_attempts = 3
context_limit_tokens = 400000
auto_compact_at_tokens = 350000
head_preserve_messages = 0
minimum_tail_messages = 0
tail_preserve_tokens = 0

[cli]
emit_compaction_status = true
show_summary_preview = false
summary_preview_chars = 160
```

Hermes config can override them:

```yaml
operational_checkpoint:
  context_limit_tokens: 400000
  auto_compact_at_tokens: 350000
  head_preserve_messages: 0
  minimum_tail_messages: 0
  tail_preserve_tokens: 0
  summary_retry_attempts: 3
  cli:
    emit_compaction_status: true
    show_summary_preview: false
    summary_preview_chars: 160
```

If you want head or tail protection, set those values explicitly. The default is full-window checkpointing because this plugin treats the compression window as the thing to preserve.

## What it looks like

During auto-compaction, you may see:

```text
🗜️  Operational Checkpoint: auto-compacting ~123,456 / 350,000 tokens...
  ✅ Operational Checkpoint reduced active context budget: ~123,456 → ~18,200 tokens
```

The wording is intentional. Message counts become misleading once checkpointing and hydration are involved. The useful number is how much active context budget was reduced.

## Tested Hermes versions

This plugin has been tested against:

- Hermes Agent v0.11.0 (2026.4.23)
- local Hermes source checkout on `main` at `9a1454060` from 2026-04-30

It may work on nearby Hermes builds, but this plugin touches runtime boundaries that can change: CLI/TUI import order, context-engine registration, manual `/compress`, auto-compaction, resume, save, hydration, and usage reporting. If those internals move, rerun the tests before trusting it.

## Quick check

Inside Hermes:

```text
/plugins
```

Make sure `operational_checkpoint` is enabled. Then try a small manual compression:

```text
/clear
say a few things
/compress
```

If the plugin is active, manual compression should run through Operational Checkpoint. Auto-compaction should use the same checkpoint path and status wording.

## Development

Run the test suite with the Hermes virtualenv:

```bash
/home/reyj94/.hermes/hermes-agent/venv/bin/python -m pytest -q
```

The tests cover activation config writing, deferred sidecar installation, TUI usage refresh, checkpoint shape validation, runtime separation, full-window compression defaults, same-session manual compression, resume hydration, and auto preflight compaction.

## License

MIT
