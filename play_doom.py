"""Play Doom autonomously with the Laya System-1 decision model.

Design: docs/agent-design.md — deterministic geometry feeds a compact state dict; one Laya
forward pass answers a tactical choice + a danger probability; deterministic safety rails
enforce alignment direction, ammo and blind-scan rules. Works across scenarios with different
button sets (turn-only, strafe-only, full movement) via capability detection.
"""

import argparse
import os
import time

import torch
import vizdoom as zd

# Laya appends the whole state as JSON next to a 192-token question head inside a 512-token
# window — keep these strings short (see docs/laya-model.md).
ALIGN_TOLERANCE_MIN = 40.0
RANGE_MELEE_PX = 60
RANGE_CLOSE_PX = 35
DANGER_TICS = 2

# Per-scenario reward threshold indicating a kill in a single make_action window (kill rewards
# are programmed in each WAD's ACS script; shot/living costs share the same window). None =
# no kill reward to detect (deadly_corridor is progress-shaped) — kills are not counted there.
KILL_THRESHOLD = {
    "defend_the_center.cfg": 1.0,
    "defend_the_line.cfg": 1.0,
    "basic.cfg": 50.0,
}
# Consecutive blind steps before the agent advances to make progress (scan-first safety).
BLIND_SCAN_BEFORE_ADVANCE = 3

# Goal thresholds shared by the priority question criteria and the rails (keep in sync).
SURVIVE_HP = 40
LOW_AMMO = 10
PANIC_HP = 25

# Goal presets per scenario: survival = engine penalty per HP lost, pressure = engine reward
# per damage dealt. Pressure stays off in kill-counted scenarios because it would corrupt
# threshold-based kill detection (both are positive rewards in the same make_action window).
DEFAULT_GOALS = {"survival": 1.0, "pressure": 0.0}
GOAL_PRESETS = {
    "defend_the_center.cfg": {"survival": 1.0, "pressure": 0.0},
    "defend_the_line.cfg": {"survival": 1.0, "pressure": 0.0},
    "basic.cfg": {"survival": 0.0, "pressure": 0.0},  # no enemy damage in this scenario
    "deadly_corridor.cfg": {"survival": 1.0, "pressure": 1.0},  # pistol: reward hits, not just kills
}


def resolve_scenario_path(scenario_name: str) -> str:
    """Find scenario config either in local folder or in ViZDoom scenarios directory."""
    if os.path.isabs(scenario_name) and os.path.exists(scenario_name):
        return scenario_name
    if os.path.exists(scenario_name):
        return os.path.abspath(scenario_name)
    builtin_path = os.path.join(zd.scenarios_path, scenario_name)
    if os.path.exists(builtin_path):
        return builtin_path
    if not scenario_name.endswith(".cfg"):
        with_cfg = os.path.join(zd.scenarios_path, f"{scenario_name}.cfg")
        if os.path.exists(with_cfg):
            return with_cfg
    raise FileNotFoundError(f"Could not find scenario: {scenario_name}")


def detect_capabilities(game: zd.DoomGame) -> dict:
    """Which action primitives the scenario's button set supports."""
    buttons = set(game.get_available_buttons())
    return {
        "turn": zd.Button.TURN_LEFT in buttons and zd.Button.TURN_RIGHT in buttons,
        "attack": zd.Button.ATTACK in buttons,
        "strafe": zd.Button.MOVE_LEFT in buttons and zd.Button.MOVE_RIGHT in buttons,
        "move": zd.Button.MOVE_FORWARD in buttons and zd.Button.MOVE_BACKWARD in buttons,
    }


def build_action_maps(game: zd.DoomGame, caps: dict) -> dict:
    """Action vectors for every tactic. Alignment uses turns when available, else strafing
    (a strafe shifts the view laterally, which centers an off-axis target)."""
    buttons = list(game.get_available_buttons())

    def idx(button):
        return buttons.index(button) if button in buttons else None

    tl, tr = idx(zd.Button.TURN_LEFT), idx(zd.Button.TURN_RIGHT)
    sl, sr = idx(zd.Button.MOVE_LEFT), idx(zd.Button.MOVE_RIGHT)
    fw, bw, atk = idx(zd.Button.MOVE_FORWARD), idx(zd.Button.MOVE_BACKWARD), idx(zd.Button.ATTACK)

    def vec(turn_l=False, turn_r=False, strafe_l=False, strafe_r=False, fwd=False, back=False, attack=False):
        v = [0] * len(buttons)
        for enabled, i in ((turn_l, tl), (turn_r, tr), (strafe_l, sl), (strafe_r, sr), (fwd, fw), (back, bw), (attack, atk)):
            if enabled and i is not None:
                v[i] = 1
        return v

    align_l = vec(turn_l=True) if caps["turn"] else vec(strafe_l=True)
    align_r = vec(turn_r=True) if caps["turn"] else vec(strafe_r=True)

    maps = {
        "hold": vec(),
        "align_left": align_l,
        "align_right": align_r,
        "scan": vec(turn_r=True) if caps["turn"] else vec(strafe_r=True),
    }
    if caps["attack"]:
        maps["attack"] = vec(attack=True)
        maps["attack_align_left"] = [a | b for a, b in zip(align_l, vec(attack=True))]
        maps["attack_align_right"] = [a | b for a, b in zip(align_r, vec(attack=True))]
    if caps["move"]:
        maps["advance"] = vec(fwd=True)
        maps["retreat"] = vec(back=True)
        if caps["attack"]:
            maps["attack_advance"] = [a | b for a, b in zip(vec(fwd=True), vec(attack=True))]
            maps["attack_retreat"] = [a | b for a, b in zip(vec(back=True), vec(attack=True))]
    return maps


def make_questions(caps: dict) -> dict:
    """Laya questions — simple, literal state->tactic mapping (see docs/agent-design.md).
    Options are filtered to what the scenario's buttons can actually execute."""
    align_verb = "rotate" if caps["turn"] else "strafe"
    align_gerund = "rotating" if caps["turn"] else "strafing"
    criteria = {}
    if caps["attack"]:
        criteria["attack"] = "target is center and ammo above 0: shoot now"
        criteria["attack_align_left"] = f"target is left and range is melee or close: shoot while {align_gerund} left"
        criteria["attack_align_right"] = f"target is right and range is melee or close: shoot while {align_gerund} right"
    criteria["align_left"] = f"target is left: {align_verb} left to aim"
    criteria["align_right"] = f"target is right: {align_verb} right to aim"
    if caps["attack"] and caps["move"]:
        criteria["attack_advance"] = "target is center and range is far: shoot while moving forward"

    return {
        "action": {
            "type": "choice",
            "instructions": "Doom combat state. Pick the immediate tactic.",
            "criteria": criteria,
        },
        "priority": {
            "type": "choice",
            "instructions": "Doom combat state. What matters most right now?",
            "criteria": {
                "survive": f"health is {SURVIVE_HP} or below and near is above 0: keep distance from monsters",
                "conserve": f"ammo is {LOW_AMMO} or below and above 0: stop firing until the target is centered",
                "engage": "target is not none: fight the visible target",
                "advance": "target is none: move toward the goal",
            },
        },
        "danger": {
            "type": "noul",
            "instructions": "Is the player in immediate danger (took_damage is true or health below 40)?",
        },
    }


def analyze_scene(game: zd.DoomGame, prev_health):
    """Perception: extract variables and geometric targeting from labels.

    Returns (state_dict, geom). state_dict is what Laya sees; geom carries the deterministic
    fallback tactic and tic budget. Returns (None, None) if the episode just ended.
    """
    state = game.get_state()
    if state is None:
        return None, None

    screen_center_x = game.get_screen_width() / 2.0
    avail_vars = game.get_available_game_variables()

    health = ammo = None
    if state.game_variables is not None:
        for i, var in enumerate(avail_vars):
            if var.name == "HEALTH":
                health = int(state.game_variables[i])
            elif var.name.startswith("AMMO"):
                ammo = int(state.game_variables[i])

    enemies = []
    for l in state.labels or []:
        # Corpses keep labels ("DeadZombieman") — never count them as threats.
        if l.object_name.lower().startswith("dead"):
            continue
        if getattr(l, "object_category", None) == "Monster":
            enemies.append(l)
        elif l.object_name != "DoomPlayer" and not any(
            ign in l.object_name.lower() for ign in ("blood", "puff", "bonus", "teleport")
        ):
            enemies.append(l)  # fallback if object_category is unpopulated

    took_damage = prev_health is not None and health is not None and health < prev_health

    # Target: balance centeredness against proximity (taller bbox = closer = more dangerous).
    # Blind is never safe: monsters can attack from outside the field of view, so "no target"
    # means scan (or reposition), not wait.
    target = "none"
    target_range = "none"
    offset_x = 0.0
    target_name = ""
    tics = 4

    if enemies:
        target_label = min(enemies, key=lambda e: abs((e.x + e.width / 2.0) - screen_center_x) - 1.5 * e.height)
        cx = target_label.x + target_label.width / 2.0
        offset_x = cx - screen_center_x
        tolerance = max(ALIGN_TOLERANCE_MIN, 0.6 * target_label.width)
        target_name = target_label.object_name

        if target_label.height > RANGE_MELEE_PX:
            target_range = "melee"
        elif target_label.height > RANGE_CLOSE_PX:
            target_range = "close"
        else:
            target_range = "far"

        if abs(offset_x) <= tolerance:
            target = "center"
        elif offset_x < 0:
            target = "left"
        else:
            target = "right"

        tics = 3 if target == "center" else (6 if abs(offset_x) > 120 else 4)

    state_dict = {
        "health": health if health is not None else "unknown",
        "ammo": ammo if ammo is not None else "unknown",
        "enemies": len(enemies),
        "near": sum(1 for e in enemies if e.height > RANGE_CLOSE_PX),
        "target": target,
        "range": target_range,
        "took_damage": took_damage,
    }

    geom = {
        "health": health,
        "ammo": ammo,
        "tics": tics,
        "offset_px": int(abs(offset_x)),
        "target_name": target_name or "-",
        "side": "left" if offset_x < 0 else "right",
    }
    return state_dict, geom


def arbitrate_priority(state_dict, geom, caps):
    """Deterministic goal arbitration. The zero-shot base model cannot arbitrate conflicting
    goals (it pattern-matches 'engage' almost every step), so the critical priorities are
    rules. The model's own priority answer is compared against this in the logs — it becomes
    a supervised training signal when we fine-tune. Returns (priority or None, source)."""
    health = geom["health"]
    # Retreat as a reaction to taking hits while already hurt — as a standing state it just
    # backs the player into a corner while monsters follow (observed in deadly_corridor).
    if (
        caps["move"]
        and health is not None
        and health <= SURVIVE_HP
        and state_dict["near"] > 0
        and state_dict["took_damage"]
    ):
        return "survive", "arb"
    ammo = state_dict["ammo"]
    if isinstance(ammo, int) and 0 < ammo <= LOW_AMMO:
        return "conserve", "arb"
    return None, None


def apply_rails(choice, state_dict, geom, action_map, caps, priority):
    """Deterministic safety rails — the model advises, these enforce (docs/agent-design.md).
    The model's priority pick acts as a veto layer on top of the tactic rails.
    Returns (choice, used_fallback)."""
    used_fallback = choice not in action_map
    if used_fallback:
        choice = "hold"

    ammo_ok = geom["ammo"] is None or geom["ammo"] > 0
    close = state_dict["range"] in ("melee", "close")

    if state_dict["target"] == "none":
        # Blind: spin to reacquire — but in movement scenarios, alternate with short advances
        # or the goal (distance-shaped reward) never gets closer. Standing still got the
        # player eaten in testing, so hold is never an option.
        blind_streak = geom["blind_streak"]
        if "advance" in action_map and blind_streak >= BLIND_SCAN_BEFORE_ADVANCE:
            choice = "advance"
        else:
            choice = "scan"
        used_fallback = True
    elif state_dict["target"] == "center":
        # Centered: shoot; advance while shooting if the target is far and movement exists.
        want = "attack_advance" if state_dict["range"] == "far" and "attack_advance" in action_map else "attack"
        if not ammo_ok or not caps["attack"]:
            want = "scan"
        if choice != want:
            used_fallback = True
        choice = want
    else:
        # Off-center: align toward the target; fire while aligning only at close range.
        # Direction is geometry's call, not the model's.
        side = geom["side"]
        fire = close and ammo_ok and caps["attack"]
        want = f"attack_align_{side}" if fire else f"align_{side}"
        if choice not in (f"align_{side}", f"attack_align_{side}"):
            used_fallback = True
            choice = want
        elif choice == f"attack_align_{side}" and not fire:
            used_fallback = True
            choice = f"align_{side}"

    # Goal vetoes: the model's priority pick reshapes the tactic.
    health = geom["health"]
    if priority == "survive" and health is not None and health <= SURVIVE_HP and state_dict["near"] > 0:
        # Overwhelmed: back away; keep firing only if the target is already centered.
        if "retreat" in action_map:
            if state_dict["target"] == "center" and ammo_ok and "attack_retreat" in action_map:
                choice = "attack_retreat"
            else:
                choice = "retreat"
            used_fallback = True
    elif (
        priority == "conserve"
        and isinstance(state_dict["ammo"], int)
        and 0 < state_dict["ammo"] <= LOW_AMMO
        and choice.startswith("attack_")
    ):
        # Low ammo: align first, fire only when the shot is guaranteed (stripped below).
        choice = choice[len("attack_"):]
        used_fallback = True

    # Panic rail: regardless of the model's priority, near-death + incoming damage means run.
    if health is not None and health <= PANIC_HP and state_dict["took_damage"] and "retreat" in action_map:
        choice = "attack_retreat" if (state_dict["target"] == "center" and ammo_ok and "attack_retreat" in action_map) else "retreat"
        used_fallback = True

    return choice, used_fallback


def make_bar(value, max_val: int = 100, length: int = 10) -> str:
    filled = max(0, min(length, int((value / max_val) * length)))
    return "#" * filled + "-" * (length - filled)


def main():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Laya System-1 decision agent playing Doom (ViZDoom).")
    parser.add_argument(
        "--scenario",
        type=str,
        default="defend_the_center.cfg",
        help="Scenario name or path (default: defend_the_center.cfg)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to play (default: 1)")
    parser.add_argument("--max-steps", type=int, default=700, help="Max decision steps per episode (default: 700)")
    parser.add_argument("--headless", action="store_true", help="Run without opening the game window")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya", help="Hugging Face model ID")
    parser.add_argument("--device", type=str, default=default_device, help="Device (cuda/cpu/mps)")
    parser.add_argument("--survival", type=float, default=None, help="Engine penalty per HP lost (default: scenario preset)")
    parser.add_argument("--pressure", type=float, default=None, help="Engine reward per damage dealt (default: scenario preset)")
    args = parser.parse_args()

    scenario_file = resolve_scenario_path(args.scenario)
    scenario_base = os.path.basename(scenario_file)
    kill_threshold = KILL_THRESHOLD.get(scenario_base)
    goals = dict(GOAL_PRESETS.get(scenario_base, DEFAULT_GOALS))
    if args.survival is not None:
        goals["survival"] = args.survival
    if args.pressure is not None:
        goals["pressure"] = args.pressure

    print("=" * 70)
    print("  LAYA SYSTEM 1 DECISION AGENT -> DOOM AUTONOMOUS COMBAT")
    print("=" * 70)
    print(f"Model ID   : {args.model}")
    print(f"Device     : {args.device}" + (f" ({torch.cuda.get_device_name(0)})" if args.device == "cuda" and torch.cuda.is_available() else ""))
    print(f"Scenario   : {scenario_base}")
    print(f"Goals      : survival={goals['survival']}, pressure={goals['pressure']}")
    print(f"Display    : {'Headless' if args.headless else 'Game window (advances per decision)'}")
    print("=" * 70)

    from laya import Agent

    agent = Agent(args.model, device=args.device)

    game = zd.DoomGame()
    game.load_config(scenario_file)
    game.set_screen_resolution(zd.ScreenResolution.RES_640X480)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(not args.headless)
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    # Goal shaping layered on top of the scenario's WAD rewards (engine knobs, 1.3.0+).
    if goals["survival"]:
        game.set_damage_taken_penalty(goals["survival"])
    if goals["pressure"]:
        game.set_damage_made_reward(goals["pressure"])
    game.init()

    caps = detect_capabilities(game)
    action_map = build_action_maps(game, caps)
    questions = make_questions(caps)
    print(f"Capabilities: " + ", ".join(k for k, v in caps.items() if v))

    print("\n[Game started] Ctrl+C in this terminal to stop early.\n")

    try:
        for ep in range(1, args.episodes + 1):
            print(f">>> BEGINNING EPISODE {ep}/{args.episodes} <<<")
            game.new_episode()
            step = 0
            total_inference_time = 0.0
            prev_health = None
            prev_ammo = None
            kills = 0
            shots = 0
            blind_streak = 0

            while step < args.max_steps:
                state_dict, geom = analyze_scene(game, prev_health)
                if state_dict is None:
                    break
                blind_streak = blind_streak + 1 if state_dict["target"] == "none" else 0
                geom["blind_streak"] = blind_streak

                # Episode-effort counters feed the priority question.
                if prev_ammo is not None and geom["ammo"] is not None and geom["ammo"] < prev_ammo:
                    shots += prev_ammo - geom["ammo"]
                prev_ammo = geom["ammo"]
                state_dict["kills"] = kills if kill_threshold else "unknown"
                state_dict["shots"] = shots if geom["ammo"] is not None else "unknown"

                t0 = time.time()
                res = agent.predict(state_dict, questions)
                dt = time.time() - t0
                total_inference_time += dt

                action_ans = res["answers"]["action"]
                choice = action_ans["choice"]
                conf = action_ans.get("confidence", 0.0)
                priority_arb, priority_src = arbitrate_priority(state_dict, geom, caps)
                model_priority = res["answers"]["priority"]["choice"]
                priority = priority_arb or model_priority
                if priority_src is None:
                    priority_src = "model"
                danger_p = res["answers"]["danger"].get("noul", 0.0)

                choice, used_fallback = apply_rails(choice, state_dict, geom, action_map, caps, priority)

                tics = DANGER_TICS if danger_p > 0.5 else geom["tics"]
                reward = game.make_action(action_map[choice], tics)
                step += 1

                # Attribute damage to the tics that just ran.
                new_state = game.get_state()
                health_now = geom["health"]
                if new_state is not None and new_state.game_variables is not None:
                    for i, var in enumerate(game.get_available_game_variables()):
                        if var.name == "HEALTH":
                            health_now = int(new_state.game_variables[i])
                hp_lost = (geom["health"] - health_now) if (health_now is not None and geom["health"] is not None) else 0
                if hp_lost > 0:
                    print(f"  [DAMAGE] Took a hit: {geom['health']} -> {health_now} HP")
                prev_health = health_now

                # Kill accounting is per-scenario (WAD kill rewards differ). The survival
                # penalty shares the same reward window, so add it back before thresholding.
                adjusted = reward + hp_lost * goals["survival"]
                if kill_threshold and adjusted >= kill_threshold:
                    new_kills = max(1, int(adjusted // kill_threshold))
                    kills += new_kills
                    print(f"  [KILL] +{new_kills} frag(s) -> total {kills} | episode reward {game.get_total_reward():.1f}")

                hp_str = f"{health_now:3d}" if health_now is not None else "  -"
                hp_bar = make_bar(health_now, 100, 8) if health_now is not None else "-" * 8
                ammo_str = f"{geom['ammo']:2d}" if geom["ammo"] is not None else " -"
                ammo_bar = make_bar(geom["ammo"], 50, 6) if geom["ammo"] is not None else "-" * 6
                src = "geo" if used_fallback else "model"
                print(
                    f"[{step:03d} | {dt * 1000:4.1f}ms] "
                    f"HP: {hp_str} {hp_bar} | "
                    f"Ammo: {ammo_str} {ammo_bar} | "
                    f"Frags: {kills:2d} | "
                    f"Danger: {danger_p:4.2f} | "
                    f"Act: {choice:s} ({conf * 100:3.0f}%, {src}) | "
                    f"Prio: {priority:s} ({priority_src}) | "
                    f"{state_dict['target']:s}/{state_dict['range']:s} {geom['offset_px']:3d}px {geom['target_name']}"
                )

                if new_state is None:
                    break

            timed_out = game.is_episode_timeout_reached()
            reward = game.get_total_reward()
            avg_lat = (total_inference_time / step * 1000) if step > 0 else 0

            print("-" * 60)
            print(f"  EPISODE {ep} COMPLETE — {'survived to timeout' if timed_out else 'episode ended'}")
            print(f"  - Frags / Kills       : {kills}")
            print(f"  - Total Reward        : {reward:.1f}")
            print(f"  - Decision Steps      : {step} ({game.get_episode_time()} tics)")
            print(f"  - Avg Inference       : {avg_lat:.2f} ms")
            print("-" * 60 + "\n")
    finally:
        game.close()

    print("Done! Doom session ended.")


if __name__ == "__main__":
    main()
