# Agent Design — Laya plays Doom

How `play_doom.py` drives `defend_the_center.cfg` with `convaiinnovations/laya`, and why each
piece is the way it is. Grounded in `docs/laya-model.md` and `docs/vizdoom-defend-the-center.md`.

## Pipeline (per decision step)

```
game state
  │
  ▼
geometry (deterministic) ──► compact state dict ──► Laya predict() ──► tactical choice
  │                                                        │
  └─ fallback tactic ◄─────── action map lookup ◄────────────┘
                    │
                    ▼
        action vector + tic budget ──► game.make_action() ──► reward (= kill signal)
```

### 1. Perception — deterministic geometry

Labels (`object_category == "Monster"`, name-based fallback) give screen-space bounding boxes.
From them:

- **Target selection**: among visible monsters, minimize `abs(offset_x) − 1.5 × height` —
  balances "close to the crosshair" against "close to the player" (taller bbox = closer).
  A distant centered monster is a worse target than a melee-range one slightly off-axis.
- **Alignment**: `offset_x = bbox_center_x − screen_center_x`, tolerance
  `max(40, 0.6 × width)` → `center / left / right`.
- **Range buckets from bbox height**: `> 60 px` melee, `> 35 px` close, else far.
- **Health/ammo** read by variable name from `game_variables`.

### 2. State serialization — compact on purpose

Laya appends the whole state as JSON with a ~320-token budget (`max_len` 512 − `head_max_len`
192). The state dict is therefore small with short literal values:

```python
{"health": 87, "ammo": 43, "enemies": 3, "near": 1, "target": "left", "range": "close", "took_damage": true, "kills": 4, "shots": 12}
```

### 3. Decision — simple questions, literal vocabulary

Laya's base checkpoint is near-chance on complex zero-shot reasoning, so the question is a
**near-literal state→tactic mapping**; each option's description echoes the state's vocabulary:

- `action` (choice): `attack` / `attack_advance` / `align_left` / `align_right` /
  `attack_align_left` / `attack_align_right` (filtered by scenario capabilities) —
  descriptions like *"target is center and ammo above 0: shoot"*.
- `priority` (choice): `survive` / `conserve` / `engage` / `advance` — what matters most
  right now; see "Goals & priorities" below.
- `danger` (noul): *"Is the player in immediate danger (took_damage is true or health below
  40)?"* → `P(true)` modulates the **tic budget**: danger → 2 tics/step (fast reactions),
  calm → 3–6 tics by angular offset (fewer, longer turns).

We deliberately avoid the `score` primitive (weakest of the three, see research notes).

### 4. Safety rails — deterministic, non-negotiable

The model advises; these rules enforce (derived from observed failure modes, see git history
and the smoke-test notes):

- **Blind → always scan.** Monsters can attack from outside the screen's ~90° FOV; standing
  still while blind got the player eaten in testing.
- **Centered target + ammo → always shoot.** The zero-shot model dithered (~13% confidence)
  on an unambiguous case.
- **Ammo == 0 → never attack** (firing is a no-op anyway).
- **Fire-while-turning only at melee/close range** (a shot 300px off-axis cannot hit;
  horizontal autoaim does not exist).
- **Unknown choice → geometric fallback tactic** (never stand still because of a
  misclassification).

### 5. Actuation and kill accounting

- `game.make_action(vec, tics)`; the **returned reward is the primary event signal**. Kill
  thresholds are per-scenario (`KILL_THRESHOLD`): the WAD scripts award different amounts
  (defend_the_center +1, basic +106 minus shot/living costs in the same window) and
  deadly_corridor has no kill reward at all (progress-shaped).
- Damage is attributed to the tics that just ran by comparing health before/after the action.
- Episode ends on death (`get_state()` → `None`) or timeout
  (`is_episode_timeout_reached()`); both are reported distinctly.

## Generalization across scenarios

The agent adapts to the scenario's button set instead of hardcoding defend_the_center:

- **Capability detection** (`detect_capabilities`): turn / attack / strafe / move, from
  `get_available_buttons()`. Action vectors, question options and rails are all built only
  from available capabilities.
- **Alignment abstraction**: with turn buttons, aligning means rotating; in strafe-only
  scenarios (`basic.cfg`) it means strafing — a strafe shifts the view laterally, which
  centers an off-axis target. Same tactic name, different vector.
- **Corpse filtering**: dead monsters keep labels (`DeadZombieman`) — excluded by name
  prefix, or the agent shoots corpses forever (observed failure).
- **Blind policy**: spin to scan; in movement scenarios, after 3 consecutive blind steps take
  one advance step, or distance-shaped rewards (deadly_corridor) never accumulate.
- **Optional variables**: scenarios without HEALTH/AMMO vars report `"unknown"` in the state
  and skip the corresponding rails.
- Verified end-to-end (2026-09-21, CUDA): defend_the_center 11 kills / conserve priority
  active at low ammo, basic solved in 2 decisions (+95 reward), deadly_corridor positive
  shaped reward with survive-priority retreats when hit at low HP.

## Goals & priorities

Before learning, the agent needs more than one implicit objective. Two mechanisms encode goals:

- **Engine reward shaping** (goal presets per scenario, CLI-overridable via `--survival` /
  `--pressure`): `damage_taken_penalty` per HP lost (*survival*) and `damage_made_reward`
  per damage dealt (*pressure*, deadly_corridor only — in kill-counted scenarios it would
  corrupt threshold-based kill detection, both being positive rewards in the same window).
- **A priority question** (`survive` / `conserve` / `engage` / `advance`) whose answer vetoes
  the tactic: `survive` backs away while firing only at centered targets, `conserve` strips
  firing from align combos when ammo is low.

**Arbitration is split deliberately.** The zero-shot base model pattern-matches `engage`
almost every step (observed: 201/201), so critical priorities are decided by a deterministic
arbiter (`arbitrate_priority`): survive = HP ≤ 40 ∧ near threat ∧ damage just taken (as a
*reaction* — as a standing state it backs the player into a corner), conserve = 0 < ammo ≤ 10.
The model's own priority answer is kept and displayed for comparison — it is the supervised
training signal for the priority question once we fine-tune on session data.

## Why not ASYNC real-time mode?

`set_ticrate` only works in ASYNC modes, where the engine runs at 35 tics/s regardless of the
model. Inference (~30–40 ms) plus Python overhead can't reliably keep 35 Hz, and missed turns
in `defend_the_center` are lethal. `PLAYER` mode (blocking `make_action`) makes every tic a
deliberate decision — slower wall-clock, correct priorities. The window renders at decision
cadence.

## Tuning knobs

| Knob | Where | Effect |
|---|---|---|
| Alignment tolerance | `analyze_scene` | Larger → more `attack`, more misses; smaller → more turning |
| Target balance (`1.5 × height`) | `analyze_scene` | Higher → prefer closest monster; lower → prefer centered |
| Danger tic budget (2) | `main` loop | Lower → faster reactions, more inference calls |
| Tics by offset (3/4/6) | `analyze_scene` | Turn granularity per decision |
| `--max-steps` | CLI | Caps decisions (episode needs ≤ ~700 for 2100 tics) |
| `--survival` / `--pressure` | CLI | Goal shaping strengths (default: scenario preset) |
| `SURVIVE_HP` / `LOW_AMMO` / `PANIC_HP` | constants | Arbiter and panic thresholds |
