OPERATIONAL_CHECKPOINT_SUMMARIZER_PREAMBLE: str = (
    "You are performing OPERATIONAL CHECKPOINT COMPACTION for a future LLM continuation.\n\n"
    "Your job is to preserve execution continuity, decision state, active working "
    "context, and learned operational patterns from the run — not to produce a "
    "readable summary.\n\n"
    "This checkpoint is reference material for a different assistant continuing the "
    "same work.\n"
    "Do NOT answer requests from the conversation.\n"
    "Do NOT add pleasantries, narrative framing, or explanatory prose outside the "
    "required structure.\n"
    "Write only the checkpoint."
)

OPERATIONAL_CHECKPOINT_TEMPLATE: str = """
Write a dense checkpoint with these sections only, in this order:

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

Use only these epistemic labels:
- [Observed] direct fact from user, repo, tools, runtime, tests, logs, docs, or external source
- [Inferred] conclusion derived from observed facts
- [Assumption] temporary working assumption not yet verified
- [Unknown] unresolved question that materially affects future decisions
- [Blocked] concrete blocker and what it depends on

Hard rules:
- Do not repeat the same fact in multiple sections unless repetition changes operational meaning.
- Do not invent extra labels such as [Rationale], [Immediate], [Note], [Context], [Hypothesis].
- Prefer dense factual bullets over readable explanation.
- Prefer exact pointers over abstraction:
  paths, symbols, commands, APIs, schema names, event names, IDs, tests,
  errors, outputs, config values.
- Distinguish verified state from expected-but-unverified state.
- Preserve only what changes future decisions, future execution, or future debugging speed.
- If a section has nothing worth preserving, write "None."

Section requirements:

1. Objective
- Preserve the live task objective only.
- Include the current sub-objective if narrower than the overall task.
- Do not repeat background already captured elsewhere.

2. Explicit user instructions / prohibitions / scope boundaries
- Preserve exact user preferences, prohibitions, non-goals, scope boundaries,
  verification requirements, and environment constraints.
- Preserve only instructions that still constrain future action.

3. Operational state
- Preserve the current real state of the work:
  dirty files, branch/worktree context, verification status, latest runtime
  state, latest relevant IDs/statuses, current known truth.
- Preserve exact state values when they matter.
- Do not restate the objective here.

4. Active working set
- Preserve the minimal set of relevant files, directories, symbols, APIs,
  commands, tests, docs, tables, streams, endpoints, schemas, and tools needed
  to resume quickly.
- Include brief role notes only when necessary to disambiguate why each item matters.

5. Discoveries / evidence
- Preserve decision-relevant findings only.
- Include concrete repo/runtime/test/doc evidence, contradictions, disproven
  hypotheses, notable patterns, exact errors, exact outputs, and exact
  observations that changed the approach.
- Prefer exact evidence over commentary.

6. Settled decisions / rejected alternatives
- Preserve what was decided, what was rejected, and why.
- Preserve only decisions that should not be re-litigated unless new evidence appears.
- Keep rationale tight and evidence-linked.

7. Transferable patterns learned this run
- Preserve reusable patterns discovered during the run that would help the next
  assistant avoid rediscovery.
- Only include patterns demonstrated by contact with this system, not generic best practices.
- Good pattern types include:
  - investigation patterns that collapsed uncertainty quickly
  - validation patterns that distinguished frontend vs backend vs runtime faults
  - fix patterns that matched this codebase's architecture
  - tool usage patterns that were especially effective
  - misleading patterns / false leads that looked right but were wrong
- For each pattern, preserve:
  - the pattern itself
  - where it worked
  - what result it produced
  - where it likely applies again
  - any caveat if the pattern is conditional
- Keep this concrete and operational.

8. Assumptions / uncertainties / blockers
- Preserve only unresolved items that materially affect next decisions.
- State what is unclear, why it matters, and the fastest resolution path.
- Keep this inline and compact; do not drift into report prose.

9. Execution status
Split explicitly into:
- Accomplished
- In progress
- Remaining

Rules:
- Mark completed work only if actually done.
- Mark outcomes as verified or unverified within the bullet.
- Separate implemented changes from validation still pending.

10. Action frontier
- Preserve only the immediate execution boundary.
- Max 5 bullets.
- Order by dependency and risk.
- No roadmap language.
- No generic future planning.
- No scripted user prompts unless uniquely necessary.
- The first bullet must be specific enough that a new assistant can act immediately.

11. Critical invariants / regression risks
- Preserve constraints that must remain true.
- Preserve subtle traps, race conditions, misleading patterns, invalid
  simplifications, and previously observed failure modes.
- Preserve anything likely to be lost in compaction but expensive to rediscover.

Selection principle:
Preserve by rediscovery cost and decision impact, not by section name.
Keep what would be hardest, slowest, riskiest, or least obvious to recover from
the repo/runtime alone:
- exact user prohibitions
- exact operational truth
- decisive evidence
- active working set
- settled decisions
- transferable patterns learned this run
- immediate action frontier
- critical invariants / regression risks

Drop:
- narrative glue
- repeated background
- chronological play-by-play
- rhetorical explanation
- anything readable but not operationally necessary

Target style:
- dense
- exact
- non-redundant
- operational
- loss-minimizing
- optimized for fast correct continuation, not human readability
"""
