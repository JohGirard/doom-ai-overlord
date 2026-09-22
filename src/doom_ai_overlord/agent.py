"""Game setup and the shared per-episode step generator.

`episode_steps` is the single pipeline everything (CLI, console app, recorders) drives —
all agent behavior flows through here.
"""

import os
import time

import numpy as np
import vizdoom as zd

from .decisions import DANGER_TICS, apply_rails, arbitrate_priority
from .perception import analyze_scene


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


def setup_game(scenario_file: str, window_visible: bool, goals: dict) -> zd.DoomGame:
    """ViZDoom game instance with labels, HUD and goal shaping configured."""
    game = zd.DoomGame()
    game.load_config(scenario_file)
    game.set_screen_resolution(zd.ScreenResolution.RES_640X480)
    game.set_labels_buffer_enabled(True)
    game.set_window_visible(window_visible)
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    # Goal shaping layered on top of the scenario's WAD rewards (engine knobs, 1.3.0+).
    if goals["survival"]:
        game.set_damage_taken_penalty(goals["survival"])
    if goals["pressure"]:
        game.set_damage_made_reward(goals["pressure"])
    game.init()
    return game


def episode_steps(game, agent, questions, action_map, caps, kill_threshold, survival, max_steps):
    """Run one episode, yielding one event dict per decision step.

    Shared by the CLI and the console app so both drive the exact same pipeline. Each event
    carries the model answers, the rail-enforced choice, counters and the post-action screen
    frame (HWC uint8, or None when the episode ended).
    """
    game.new_episode()
    step = 0
    prev_health = None
    prev_ammo = None
    kills = 0
    shots = 0
    blind_streak = 0

    while step < max_steps:
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
        prev_health = health_now

        # Kill accounting is per-scenario (WAD kill rewards differ). The survival
        # penalty shares the same reward window, so add it back before thresholding.
        adjusted = reward + hp_lost * survival
        prev_kills = kills
        if kill_threshold and adjusted >= kill_threshold:
            kills += max(1, int(adjusted // kill_threshold))

        frame = None
        if new_state is not None and new_state.screen_buffer is not None:
            frame = np.ascontiguousarray(np.transpose(new_state.screen_buffer, (1, 2, 0)))

        yield {
            "step": step,
            "dt_ms": dt * 1000,
            "state_dict": state_dict,
            "geom": geom,
            "choice": choice,
            "conf": conf,
            "probs": action_ans.get("probabilities", {}),
            "priority": priority,
            "priority_src": priority_src,
            "danger_p": danger_p,
            "used_fallback": used_fallback,
            "tics": tics,
            "reward": reward,
            "hp_lost": hp_lost,
            "health": health_now,
            "kills": kills,
            "kills_delta": kills - prev_kills,
            "shots": shots,
            "total_reward": game.get_total_reward(),
            "episode_tics": game.get_episode_time(),
            "alive": new_state is not None,
            "frame": frame,
        }

        if new_state is None:
            break
