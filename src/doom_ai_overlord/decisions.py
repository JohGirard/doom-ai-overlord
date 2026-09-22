"""Decision layer: capabilities, action maps, Laya questions, goal arbiter, safety rails.

The model advises; the rails enforce. Question wording mirrors the state's literal
vocabulary on purpose (the zero-shot base model cannot do free-form reasoning, and Laya
does no templating — see docs/laya-model.md).
"""

import vizdoom as zd

DANGER_TICS = 2

# Goal thresholds shared by the priority question criteria and the rails (keep in sync).
SURVIVE_HP = 40
LOW_AMMO = 10
PANIC_HP = 25

# Per-scenario reward threshold indicating a kill in a single make_action window (kill rewards
# are programmed in each WAD's ACS script; shot/living costs share the same window). None =
# no kill reward to detect (deadly_corridor is progress-shaped) — kills are not counted there.
KILL_THRESHOLD = {
    "defend_the_center.cfg": 1.0,
    "defend_the_line.cfg": 1.0,
    "basic.cfg": 50.0,
}

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
        if "advance" in action_map and blind_streak >= 3:
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
