# Contributing

Thanks for your interest — this project is a test bench, and the test matrix is huge. Every
contribution (scenario, behavior, stat, or docs) is welcome.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+. An NVIDIA GPU is recommended
(torch is pinned to the CUDA 12.6 build; CPU works, just slower).

```bash
git clone https://github.com/JohGirard/doom-ai-overlord.git
cd doom-ai-overlord
uv sync
uv run doom-agent --headless --episodes 1 --max-steps 60   # smoke test
```

The Laya model (~850 MB) downloads from Hugging Face on first run and is then used offline.

## How the pieces fit

Read these before changing behavior — they explain *why* the code is the way it is:

- `docs/agent-design.md` — the perception → decision → actuation pipeline, goals and rails.
- `docs/laya-model.md` — Laya's exact API and its quirks (no templating, token budget,
  confidence ≠ accuracy).
- `docs/vizdoom-defend-the-center.md` — scenario mechanics and reward structure.

Quick map of the package (all under `src/doom_ai_overlord/`):

| Module | Role |
|---|---|
| `perception.py` | `analyze_scene`: labels → target, range, health/ammo (deterministic) |
| `decisions.py` | `make_questions` (Laya questions), `arbitrate_priority`, `apply_rails`, capability/action-map builders, goal presets |
| `agent.py` | `setup_game`, `episode_steps` — one-episode generator; **all behaviors flow through here** |
| `app.py` | console UI + recorders; consumes `episode_steps` and never re-implements agent logic |
| `hud.py` | HUD overlay + hardware info for recordings |

## Ground rules for behavior changes

1. **The model advises; rails enforce.** New behaviors should follow that split: expose the
   information in the state dict, ask a simple literal question, and add a deterministic
   guard for the failure mode you're fixing.
2. **Keep the state dict compact** — Laya serializes it as JSON in a ~320-token budget
   (`docs/laya-model.md`).
3. **Don't trust raw confidence.** It is entropy, not accuracy. Never gate correctness on it.

## Verifying a change

Behavior changes must include an episode summary as proof (this is also the PR template):

```bash
uv run doom-agent --headless --scenario defend_the_center --episodes 1
```

Single episodes have high variance — for anything touching tactics, run at least
`--episodes 3` and report the spread, not just the best run.

## Ideas where help is wanted

See the [issue tracker](https://github.com/JohGirard/doom-ai-overlord/issues) — issues
labeled `good first issue` are self-contained and well-scoped.

## Pull requests

- One logical change per PR, with the verification summary filled in.
- The repo protects `main` (no force pushes); PRs merge into `main`.
- No code formatter is enforced; match the surrounding style and keep functions small.
