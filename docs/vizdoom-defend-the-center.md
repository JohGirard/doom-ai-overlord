# ViZDoom `defend_the_center` — Mechanics Reference

Verified against the installed `vizdoom 1.3.1` package (`.venv/Lib/site-packages/vizdoom/`)
and one full headless episode run. Scenario file:
`vizdoom/scenarios/defend_the_center.cfg`.

## Scenario rules

| Setting | Value |
|---|---|
| Map | Large circular arena; the player stands at the exact center |
| Monsters | 5 melee-only monsters spawn on the walls, respawn after dying |
| Buttons (in order) | `TURN_LEFT`, `TURN_RIGHT`, `ATTACK` — no movement, no freelook |
| Game variables (in order) | `AMMO2` (shells, start = 50), `HEALTH` |
| `doom_skill` | 3 (Hurt Me Plenty) |
| `episode_start_time` | 10 tics (skips the unholster animation) |
| `episode_timeout` | 2100 tics (60 s at 35 tics/s) |
| `death_penalty` | 1 (engine knob) |
| Kill reward | **+1.0 per kill, programmed in the WAD's ACS script — NOT the engine `kill_reward`** (which reads 0) |
| Ammo penalty | **None.** "wasting ammunition is not very good" in the docs is narrative only |

Empirically confirmed episode: 15 kills then death at tic ~818 → total reward 14.0,
exactly matching `+1/kill − 1/death`.

Consequences:

- **Kill detection**: `make_action()` returns the reward summed over the tics it processed —
  `+1.0` (or `+2.0` for two kills in one window) is the reliable kill signal.
  `GameVariable.KILLCOUNT` is unreliable in this scenario; do not use it.
- **Optimal policy**: stand still, keep turning toward the nearest monster, fire only when
  roughly centered. Moving is impossible; every miss is just lost time (and there is no ammo
  penalty, so holding fire while aligning is nearly free).
- **Death is inevitable**: 50 shells vs. endlessly respawning monsters. "Playing correctly"
  means maximizing kills before death, not surviving.

## Python API facts that matter

- `make_action(action, tics) -> float`: the action vector is held for all `tics` (1 tic =
  1/35 s), the engine processes them, and the summed reward is returned. Vector length =
  `get_available_buttons_size()`, entries in cfg button order.
- **Buttons compose**: `[0, 1, 1]` = turn right *while* firing. Multi-button vectors are the
  intended design (Gymnasium `MultiBinary` action spaces exist because of this).
- `get_state()` returns `None` once the episode is finished. `GameState.labels` is only
  populated with `set_labels_buffer_enabled(True)` (done before `init()`).
- `Label` fields: `x, y, width, height` (screen-space bbox), `object_name` (`DoomPlayer`,
  monster names), `object_category` — **`"Monster"` is a real category** in 1.3.0
  (`vizdoom.get_default_categories()` lists it), auto-assigned from engine object types.
- `is_player_dead()`, `is_episode_timeout_reached()`, `get_episode_time()`,
  `get_total_reward()`, `get_last_reward()` all exist.
- `get_game_variable(var)` works even for variables not in `available_game_variables`.

## Real-time display correction

The old script called `game.set_ticrate(35)` for "live 35 FPS rendering" — **`set_ticrate` only
affects ASYNC modes and is a no-op in the default `PLAYER` mode** (stated verbatim in the stub
docs). In `PLAYER` mode the engine advances *only* inside `make_action`/`advance_action`, so the
window renders one decision step per call. To get smooth real-time play you must either accept
decision-cadence rendering (small `tics` per step, inference-paced) or switch to
`Mode.ASYNC_PLAYER` + `set_ticrate(35)` + `set_action`/`advance_action` (the model then has to
keep up with the engine in real time — worse for correctness). We keep `PLAYER` mode.

## Neighboring scenarios (for later)

| Scenario | Buttons | Reward notes |
|---|---|---|
| `basic.cfg` | MOVE_LEFT/RIGHT, ATTACK | `living_reward=-1`, +106/kill, −5/shot (WAD) |
| `deadly_corridor.cfg` | + MOVE_FWD/BACK | `death_penalty=100`, shaped by distance to vest |
| `rocket_basic.cfg` | MOVE_LEFT/RIGHT, ATTACK | `living_reward=-1`, `sv_noautoaim 1` |
