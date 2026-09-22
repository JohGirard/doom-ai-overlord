"""Perception: screen labels and game variables -> compact state + geometry.

Deterministic and model-free — this is what the agent *sees*. The state dict produced here
is what Laya reads (keep it small: ~320 JSON tokens, see docs/laya-model.md).
"""

import vizdoom as zd

ALIGN_TOLERANCE_MIN = 40.0
RANGE_MELEE_PX = 60
RANGE_CLOSE_PX = 35


def analyze_scene(game: zd.DoomGame, prev_health):
    """Extract variables and geometric targeting from labels.

    Returns (state_dict, geom). state_dict is what Laya sees; geom carries deterministic
    details for the rails and the UI. Returns (None, None) if the episode just ended.
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
