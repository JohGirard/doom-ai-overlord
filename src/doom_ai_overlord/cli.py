"""Terminal CLI: run episodes and watch the agent decide, line by line."""

import argparse
import os

import torch

from .agent import episode_steps, resolve_scenario_path, setup_game
from .decisions import (
    DEFAULT_GOALS,
    GOAL_PRESETS,
    KILL_THRESHOLD,
    build_action_maps,
    detect_capabilities,
    make_questions,
)


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

    game = setup_game(scenario_file, window_visible=not args.headless, goals=goals)

    caps = detect_capabilities(game)
    action_map = build_action_maps(game, caps)
    questions = make_questions(caps)
    print("Capabilities: " + ", ".join(k for k, v in caps.items() if v))

    print("\n[Game started] Ctrl+C in this terminal to stop early.\n")

    try:
        for ep in range(1, args.episodes + 1):
            print(f">>> BEGINNING EPISODE {ep}/{args.episodes} <<<")
            total_inference_time = 0.0
            step = 0
            ev = None

            for ev in episode_steps(game, agent, questions, action_map, caps, kill_threshold, goals["survival"], args.max_steps):
                total_inference_time += ev["dt_ms"]
                step = ev["step"]

                if ev["kills_delta"] > 0:
                    print(f"  [KILL] +{ev['kills_delta']} frag(s) -> total {ev['kills']} | episode reward {ev['total_reward']:.1f}")
                if ev["hp_lost"] > 0:
                    print(f"  [DAMAGE] Took a hit: {ev['geom']['health']} -> {ev['health']} HP")

                health_now = ev["health"]
                geom = ev["geom"]
                hp_str = f"{health_now:3d}" if health_now is not None else "  -"
                hp_bar = make_bar(health_now, 100, 8) if health_now is not None else "-" * 8
                ammo_str = f"{geom['ammo']:2d}" if geom["ammo"] is not None else " -"
                ammo_bar = make_bar(geom["ammo"], 50, 6) if geom["ammo"] is not None else "-" * 6
                src = "geo" if ev["used_fallback"] else "model"
                sd = ev["state_dict"]
                print(
                    f"[{step:03d} | {ev['dt_ms']:4.1f}ms] "
                    f"HP: {hp_str} {hp_bar} | "
                    f"Ammo: {ammo_str} {ammo_bar} | "
                    f"Frags: {ev['kills']:2d} | "
                    f"Danger: {ev['danger_p']:4.2f} | "
                    f"Act: {ev['choice']:s} ({ev['conf'] * 100:3.0f}%, {src}) | "
                    f"Prio: {ev['priority']:s} ({ev['priority_src']}) | "
                    f"{sd['target']:s}/{sd['range']:s} {geom['offset_px']:3d}px {geom['target_name']}"
                )

            timed_out = game.is_episode_timeout_reached()
            reward = game.get_total_reward()
            avg_lat = (total_inference_time / step * 1000) if step > 0 else 0

            print("-" * 60)
            print(f"  EPISODE {ep} COMPLETE — {'survived to timeout' if timed_out else 'episode ended'}")
            print(f"  - Frags / Kills       : {ev['kills'] if ev else 0}")
            print(f"  - Total Reward        : {reward:.1f}")
            print(f"  - Decision Steps      : {step} ({game.get_episode_time()} tics)")
            print(f"  - Avg Inference       : {avg_lat:.2f} ms")
            print("-" * 60 + "\n")
    finally:
        game.close()

    print("Done! Doom session ended.")


if __name__ == "__main__":
    main()
