"""Console app: watch the agent play in a terminal-style window, and record demos.

The whole UI is drawn on one Tk canvas: monospace phosphor-green on near-black, dotted game
viewport, model probabilities, estimates, performance and rails counters. Also records a
shareable GIF of the app itself (--capture) and an H.264 gameplay video (--record).

Keys: SPACE pause · R reset episode · Q quit
"""

import argparse
import os
import queue
import threading
import time
from collections import deque

import numpy as np
import torch
from PIL import Image, ImageDraw

from .agent import episode_steps, resolve_scenario_path, setup_game
from .decisions import (
    DEFAULT_GOALS,
    GOAL_PRESETS,
    KILL_THRESHOLD,
    build_action_maps,
    detect_capabilities,
    make_questions,
)
from .hud import hardware_info, hud_overlay

# ---------------------------------------------------------------- terminal palette
BG = "#0a0e0b"
BORDER = "#1e2b21"
FAINT = "#16211a"
DIM = "#4b5f51"
LABEL = "#6f8a76"
TXT = "#c9dccf"
GREEN = "#56e39f"
GREEN_DIM = "#2f8f5c"
WARN = "#e3c46a"
RED = "#e56b6b"

GAME_X, GAME_Y, GAME_W, GAME_H = 36, 124, 520, 390
RIGHT_X, RIGHT_W = 600, 446
SPARK_LEN = 120

TACTIC_ORDER = ["attack", "attack_advance", "attack_align_left", "attack_align_right",
                "align_left", "align_right", "advance", "retreat", "scan", "hold"]
PERF_ORDER = ["INFERENCE", "P95 RESPONSE", "DECISIONS", "ENGINE", "DEVICE", "NETWORK"]

F = lambda s, b=False: ("Consolas", s, "bold") if b else ("Consolas", s)


def run_pipeline(args, gui_queue=None, writer=None, stop=None, cmd_queue=None):
    scenario_file = resolve_scenario_path(args.scenario)
    scenario_base = os.path.basename(scenario_file)
    kill_threshold = KILL_THRESHOLD.get(scenario_base)
    goals = dict(GOAL_PRESETS.get(scenario_base, DEFAULT_GOALS))
    if getattr(args, "survival", None) is not None:
        goals["survival"] = args.survival
    if getattr(args, "pressure", None) is not None:
        goals["pressure"] = args.pressure

    from laya import Agent

    agent = Agent(args.model, device=args.device)
    game = setup_game(scenario_file, window_visible=False, goals=goals)
    caps = detect_capabilities(game)
    action_map = build_action_maps(game, caps)
    questions = make_questions(caps)
    hw = hardware_info(args, scenario_base, goals)
    hw_line = f"{hw['GPU']} · torch {hw['torch']} · System-1 decision model"

    if gui_queue is not None:
        gui_queue.put(("hw", {"hw": hw, "tactics": [t for t in TACTIC_ORDER if t in action_map]}, None))

    flashes = []
    frame_idx = 0
    ep = 0
    t_start = time.time()

    def drain_commands():
        """Returns 'reset' / 'quit' / None. Blocks while paused."""
        if cmd_queue is None:
            return None
        while True:
            try:
                c = cmd_queue.get_nowait()
            except queue.Empty:
                return None
            if c == "pause":
                gui_queue.put(("paused", True, None))
                while True:
                    c2 = cmd_queue.get()
                    if c2 == "resume":
                        gui_queue.put(("paused", False, None))
                        break
                    if c2 in ("reset", "quit"):
                        gui_queue.put(("paused", False, None))
                        return c2
            elif c in ("reset", "quit"):
                return c

    try:
        while stop is None or not stop.is_set():
            ep += 1
            reset = False
            for ev in episode_steps(game, agent, questions, action_map, caps,
                                    kill_threshold, goals["survival"], args.max_steps):
                cmd = drain_commands()
                if cmd == "quit":
                    stop.set()
                    break
                if cmd == "reset":
                    reset = True
                    break
                frame_idx += 1
                if ev["kills_delta"] > 0:
                    flashes.append((f"KILL +{ev['kills_delta']}  (total {ev['kills']})",
                                    (86, 227, 159, 255), frame_idx + 18))
                if ev["hp_lost"] > 0:
                    flashes.append((f"-{ev['hp_lost']} HP", (229, 107, 107, 255), frame_idx + 18))
                flashes = [f for f in flashes if f[2] > frame_idx]

                if ev["frame"] is not None:
                    if writer is not None:
                        writer.write(hud_overlay(ev["frame"], ev, hw_line, flashes))
                    if gui_queue is not None:
                        ev2 = dict(ev)
                        ev2["episode"] = ep
                        item = ("ev", ev2, Image.fromarray(ev["frame"]))
                        try:
                            gui_queue.put_nowait(item)
                        except queue.Full:
                            try:
                                gui_queue.get_nowait()
                            except queue.Empty:
                                pass
                            gui_queue.put_nowait(item)

                if (stop is not None and stop.is_set()) or (args.seconds and time.time() - t_start > args.seconds):
                    break

            if not reset and (stop is not None and not stop.is_set()):
                if gui_queue is not None:
                    gui_queue.put(("summary", {
                        "episode": ep, "reward": game.get_total_reward(),
                        "tics": game.get_episode_time(),
                        "timeout": game.is_episode_timeout_reached(),
                    }, None))
            if stop is not None and stop.is_set():
                break
            if reset:
                continue
            if args.episodes and ep >= args.episodes:
                break
            if args.seconds and time.time() - t_start > args.seconds:
                break
    finally:
        game.close()


# ---------------------------------------------------------------- demo recording

def run_record(args):
    import subprocess

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "640x480", "-r", str(args.fps), "-i", "-",
           "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium",
           args.record]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    class Writer:
        def write(self, img):
            proc.stdin.write(np.asarray(img).tobytes())

        def release(self):
            proc.stdin.close()
            proc.wait()

    writer = Writer()
    try:
        run_pipeline(args, writer=writer)
    finally:
        writer.release()
    size_mb = os.path.getsize(args.record) / 1e6
    print(f"Saved {args.record} ({size_mb:.1f} MB, {args.fps} fps, H.264)")

    if args.poster:
        subprocess.run([ffmpeg, "-y", "-ss", str(args.poster_at), "-i", args.record,
                        "-frames:v", "1", args.poster],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Saved poster {args.poster}")


# ---------------------------------------------------------------- console UI

def dotted_bg(width, height, step=16, dot="#14201a", bg="#05080a"):
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)
    for y in range(step // 2, height, step):
        for x in range(step // 2, width, step):
            d.point((x, y), fill=dot)
    return ImageTk.PhotoImage(img)


def run_app(args):
    global tk, ImageTk
    import tkinter as tk
    from PIL import ImageTk

    W, H = 1080, 640
    stop = threading.Event()
    q = queue.Queue(maxsize=4)
    cmds = queue.Queue()
    threading.Thread(target=run_pipeline, args=(args, q, None, stop, cmds), daemon=True).start()

    root = tk.Tk()
    root.title("doom-ai-overlord — local intelligence")
    root.configure(bg=BG)
    root.resizable(False, False)

    cv = tk.Canvas(root, width=W, height=H, bg=BG, highlightthickness=0)
    cv.pack()

    def line(x1, y1, x2, y2, color=FAINT, w=1):
        return cv.create_line(x1, y1, x2, y2, fill=color, width=w)

    def text(x, y, s, size=10, color=TXT, bold=False, anchor="w"):
        return cv.create_text(x, y, text=s, font=F(size, bold), fill=color, anchor=anchor)

    def meter_row(y, label):
        lbl = text(RIGHT_X, y, label, 10, LABEL)
        cv.create_rectangle(760, y - 6, 940, y + 6, outline="", fill=FAINT)
        fill = cv.create_rectangle(760, y - 6, 760, y + 6, outline="", fill=GREEN_DIM)
        val = text(RIGHT_X + RIGHT_W, y, "0.00", 10, TXT, anchor="e")
        return lbl, fill, val

    # ---------------- chrome
    cv.create_rectangle(10, 10, W - 10, H - 10, outline=BORDER)
    cv.create_rectangle(14, 14, W - 14, H - 14, outline=FAINT)
    for i, c in enumerate(("#d4736c", "#d9ae4e", "#58b368")):
        cv.create_oval(28 + i * 18, 18, 36 + i * 18, 26, fill=c, outline="")
    text(W / 2, 22, "doom-ai-overlord  /  live agent decisions", 10, DIM, anchor="center")
    text(W - 34, 22, "LIVE RUN  ·  1×", 10, GREEN, anchor="e")

    text(34, 62, "DOOM AI OVERLORD  /  SYSTEM-1 AGENT", 13, TXT)
    ep_lbl = text(W - 34, 62, "EPISODE 01", 12, GREEN, anchor="e")
    line(34, 84, W - 34, 84)

    # ---------------- left: game viewport
    scen_lbl = text(36, 108, "SCENARIO  —", 10, LABEL)
    tics_lbl = text(556, 108, "", 10, LABEL, anchor="e")
    cv.create_rectangle(GAME_X, GAME_Y, GAME_X + GAME_W, GAME_Y + GAME_H, outline="#2f5240")
    cv.create_image(GAME_X + 1, GAME_Y + 1, image=dotted_bg(GAME_W - 2, GAME_H - 2), anchor="nw")
    frame_item = cv.create_image(GAME_X + GAME_W / 2, GAME_Y + GAME_H / 2, anchor="center")
    flash_lbl = text(GAME_X + GAME_W / 2, GAME_Y + 26, "", 16, GREEN, bold=True, anchor="center")

    big = {}
    for name, x, col in (("FRAGS", 36, GREEN), ("HEALTH", 180, TXT), ("AMMO", 324, TXT), ("REWARD", 468, TXT)):
        text(x, 538, name, 9, LABEL)
        big[name] = text(x, 566, "0", 26, col, bold=True)
    cv.create_rectangle(36, 596, 556, 598, outline="", fill=FAINT)
    hp_line = cv.create_rectangle(36, 596, 36, 598, outline="", fill=GREEN_DIM)
    hp_pct = text(556, 597, "", 9, DIM, anchor="e")

    text(36, 622, "SPACE pause      R reset      Q quit", 10, DIM)
    runtimer = text(W - 34, 622, "ESTIMATES BY LAYA      00:00", 10, DIM, anchor="e")

    # ---------------- right: estimates panel (rows are built when the agent comes online)
    ui = {}
    prob_rows = {}
    prob_items = []

    def build_right(tactics, hw):
        for item_id in prob_items:
            cv.delete(item_id)
        prob_rows.clear()
        del prob_items[:]

        def reg(item_id):
            prob_items.append(item_id)
            return item_id

        y = 178
        for name in tactics:
            row = {
                "mark": reg(text(RIGHT_X, y, "", 10, GREEN, bold=True)),
                "name": reg(text(RIGHT_X + 18, y, name.upper(), 10, LABEL)),
                "fill": reg(cv.create_rectangle(760, y - 6, 760, y + 6, outline="", fill=GREEN_DIM)),
                "val": reg(text(RIGHT_X + RIGHT_W, y, "0.00", 10, LABEL, anchor="e")),
            }
            reg(cv.create_rectangle(760, y - 6, 940, y + 6, outline="", fill=FAINT))
            cv.tag_raise(row["fill"])
            prob_rows[name] = row
            y += 24

        y += 14
        ui["exec_lbl"] = reg(text(RIGHT_X, y, "EXECUTING", 10, LABEL))
        ui["exec_val"] = reg(text(RIGHT_X + 110, y, "", 13, GREEN, bold=True))
        ui["exec_src"] = reg(text(RIGHT_X + RIGHT_W, y, "", 9, DIM, anchor="e"))
        y += 30
        _lbl, ui["danger_fill"], ui["danger_val"] = meter_row(y, "DANGER ESTIMATE")
        reg(_lbl)
        y += 26
        ui["prio_lbl"] = reg(text(RIGHT_X, y, "PRIORITY", 10, LABEL))
        ui["prio_val"] = reg(text(RIGHT_X + 110, y, "", 11, TXT, bold=True))
        ui["prio_src"] = reg(text(RIGHT_X + RIGHT_W, y, "", 9, DIM, anchor="e"))
        y += 24
        reg(line(RIGHT_X, y, RIGHT_X + RIGHT_W, y))
        y += 24
        defaults = {"ENGINE": f"torch {hw['torch']}  ·  FP16", "DEVICE": hw["GPU"],
                    "NETWORK": "OFFLINE  ·  cached"}
        for name in PERF_ORDER:
            ui[f"perf_lbl_{name}"] = reg(text(RIGHT_X, y, name, 10, LABEL))
            ui[f"perf_val_{name}"] = reg(text(RIGHT_X + RIGHT_W, y, defaults.get(name, "—"),
                                              10, GREEN, anchor="e"))
            y += 21
        y += 10
        ui["rails_lbl"] = reg(text(RIGHT_X, y, "Laya + safety rails", 10, "#b8d9c2"))
        ui["rails_val"] = reg(text(RIGHT_X + RIGHT_W, y, "rails interventions   0000",
                                   10, GREEN, anchor="e"))

    model_hdr = text(RIGHT_X, 108, "Laya System-1", 11, GREEN, bold=True)
    model_sub = text(RIGHT_X, 126, "connecting…", 9, DIM)
    text(RIGHT_X, 156, "NEXT ACTION", 10, LABEL)
    text(RIGHT_X + RIGHT_W, 156, "MODEL PROBABILITIES", 10, LABEL, anchor="e")

    paused_rect = cv.create_rectangle(GAME_X, GAME_Y, GAME_X + GAME_W, GAME_Y + GAME_H,
                                      fill="#05080a", stipple="gray50", outline="")
    paused_txt = text(GAME_X + GAME_W / 2, GAME_Y + GAME_H / 2, "PAUSED", 22, GREEN,
                      bold=True, anchor="center")
    cv.itemconfigure(paused_rect, state="hidden")
    cv.itemconfigure(paused_txt, state="hidden")

    # ---------------- window capture (records the console itself, GIF via ffmpeg)
    capture = getattr(args, "capture", None)
    cap_proc, cap_bbox, cap_until, cap_last = None, None, None, 0.0
    if capture:
        import subprocess

        import imageio_ffmpeg

        root.overrideredirect(True)  # frameless: the capture is the app, not the desktop
        root.geometry("+80+80")
        root.update_idletasks()
        cap_bbox = (root.winfo_rootx(), root.winfo_rooty(),
                    root.winfo_rootx() + W, root.winfo_rooty() + H)
        fps = 8
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-an", "-vf", "fps=8,split[s0][s1];[s0]palettegen=max_colors=160[p];[s1][p]paletteuse",
               capture]
        cap_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print(f"capture: window at {cap_bbox}, writing {capture} ({fps} fps gif)")

    # ---------------- state
    latencies = deque(maxlen=SPARK_LEN)
    stamps = deque(maxlen=400)
    state = {"interventions": 0, "t0": None, "flash": ("", 0.0)}

    def percentile(values, p):
        if not values:
            return 0.0
        s = sorted(values)
        return s[min(len(s) - 1, int(p * (len(s) - 1)))]

    def on_event(ev, img):
        if state["t0"] is None:
            state["t0"] = time.time()
        if img is not None:
            photo = ImageTk.PhotoImage(img.resize((GAME_W - 4, GAME_H - 4), Image.NEAREST))
            cv.itemconfigure(frame_item, image=photo)
            cv.frame_photo = photo
        latencies.append(ev["dt_ms"])
        stamps.append(time.time())
        now = time.time()
        while stamps and now - stamps[0] > 10:
            stamps.popleft()
        if ev["used_fallback"]:
            state["interventions"] += 1

        hp = ev["health"]
        cv.itemconfigure(big["HEALTH"], text="-" if hp is None else str(hp))
        frac = 0.0 if hp is None else max(0.0, min(1.0, hp / 100.0))
        cv.coords(hp_line, 36, 596, 36 + int(520 * frac), 598)
        cv.itemconfigure(hp_line, fill=GREEN_DIM if frac > 0.25 else RED)
        cv.itemconfigure(hp_pct, text=f"{frac * 100:.1f}%")
        cv.itemconfigure(big["FRAGS"], text=str(ev["kills"]))
        cv.itemconfigure(big["REWARD"], text=f"{ev['total_reward']:.0f}")
        cv.itemconfigure(big["AMMO"], text="-" if ev["geom"]["ammo"] is None else str(ev["geom"]["ammo"]))
        cv.itemconfigure(tics_lbl, text=f"{ev['episode_tics']} tics  ·  step {ev['step']}")
        cv.itemconfigure(ep_lbl, text=f"EPISODE {ev['episode']:02d}")

        for name, row in prob_rows.items():
            p = ev["probs"].get(name, 0.0)
            executing = name == ev["choice"]
            cv.itemconfigure(row["mark"], text=">" if executing else "")
            cv.itemconfigure(row["name"], fill=GREEN if executing else LABEL,
                             font=F(10, executing))
            x1, y1, _x2, y2 = cv.coords(row["fill"])
            cv.coords(row["fill"], x1, y1, 760 + int(180 * p), y2)
            cv.itemconfigure(row["fill"], fill=GREEN if executing else GREEN_DIM)
            cv.itemconfigure(row["val"], text=f"{p:.2f}", fill=TXT if executing else LABEL)

        src = "rails" if ev["used_fallback"] else "model"
        cv.itemconfigure(ui["exec_val"], text=ev["choice"].upper())
        cv.itemconfigure(ui["exec_src"], text=f"via {src}  ·  {ev['conf'] * 100:.0f}% conf")
        dp = ev["danger_p"]
        x1, y1, _x2, y2 = cv.coords(ui["danger_fill"])
        cv.coords(ui["danger_fill"], x1, y1, 760 + int(180 * dp), y2)
        cv.itemconfigure(ui["danger_fill"], fill=RED if dp > 0.5 else GREEN_DIM)
        cv.itemconfigure(ui["danger_val"], text=f"{dp:.2f}")
        cv.itemconfigure(ui["prio_val"], text=ev["priority"].upper())
        cv.itemconfigure(ui["prio_src"], text=f"via {ev['priority_src']}")

        vals = list(latencies)
        avg = sum(vals) / len(vals)
        dps = len(stamps) / 10.0
        cv.itemconfigure(ui["perf_val_INFERENCE"], text=f"{avg:5.1f} ms avg")
        cv.itemconfigure(ui["perf_val_P95 RESPONSE"], text=f"{percentile(vals, 0.95):5.1f} ms")
        cv.itemconfigure(ui["perf_val_DECISIONS"], text=f"{dps:4.1f} /s")
        cv.itemconfigure(ui["rails_val"],
                         text=f"rails interventions   {state['interventions']:04d}")

        flash_text, flash_until = state["flash"]
        if ev["kills_delta"] > 0:
            flash_text, flash_until = f"KILL +{ev['kills_delta']}   total {ev['kills']}", now + 1.6
        elif ev["hp_lost"] > 0:
            flash_text, flash_until = f"-{ev['hp_lost']} HP", now + 1.2
        state["flash"] = (flash_text, flash_until)
        if now < flash_until:
            color = GREEN if flash_text.startswith("KILL") else RED
            cv.itemconfigure(flash_lbl, text=flash_text, fill=color)
        else:
            cv.itemconfigure(flash_lbl, text="")

        elapsed = int(now - state["t0"])
        cv.itemconfigure(runtimer,
                         text=f"ESTIMATES BY LAYA      {elapsed // 60:02d}:{elapsed % 60:02d}")

    def poll():
        nonlocal cap_until, cap_last
        got_ev = False
        try:
            while True:
                kind, payload, img = q.get_nowait()
                if kind == "hw":
                    hw = payload["hw"]
                    cv.itemconfigure(model_hdr, text="Laya System-1")
                    cv.itemconfigure(model_sub, text=f"{hw['model']}  ·  local")
                    cv.itemconfigure(scen_lbl, text=f"SCENARIO  {hw['scenario']}")
                    build_right(payload["tactics"], hw)
                elif kind == "ev":
                    got_ev = True
                    on_event(payload, img)
                elif kind == "paused":
                    st = "normal" if payload else "hidden"
                    cv.itemconfigure(paused_rect, state=st)
                    cv.itemconfigure(paused_txt, state=st)
        except queue.Empty:
            pass

        # capture: arm when the model is loaded (first gameplay event), then grab the
        # frameless window at ~8 fps until the clip length is reached
        if cap_proc is not None and got_ev and cap_until is None:
            cap_until = time.time() + getattr(args, "capture_seconds", 32)
            print("capture: model loaded — recording")
        if cap_proc is not None and cap_until is not None:
            now = time.time()
            if now >= cap_until:
                cap_proc.stdin.close()
                cap_proc.wait()
                print(f"capture saved: {capture} ({os.path.getsize(capture) / 1e6:.1f} MB)")
                quit_()
                return
            if now - cap_last >= 0.12:
                cap_last = now
                from PIL import ImageGrab
                grab = ImageGrab.grab(bbox=cap_bbox).convert("RGB")
                cap_proc.stdin.write(np.asarray(grab).tobytes())

        if not stop.is_set():
            root.after(60, poll)

    def toggle_pause(_=None):
        paused = getattr(toggle_pause, "paused", False)
        toggle_pause.paused = not paused
        cmds.put("resume" if paused else "pause")

    def reset(_=None):
        toggle_pause.paused = False
        cmds.put("resume")
        cmds.put("reset")

    def quit_(_=None):
        stop.set()
        cmds.put("quit")
        root.destroy()

    root.bind("<space>", toggle_pause)
    root.bind("r", reset)
    root.bind("q", quit_)
    root.protocol("WM_DELETE_WINDOW", quit_)
    poll()
    root.mainloop()


def main():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Doom AI Overlord — local intelligence console.")
    parser.add_argument("--scenario", type=str, default="defend_the_center.cfg")
    parser.add_argument("--episodes", type=int, default=0, help="Episodes to run (0 = loop until closed)")
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--model", type=str, default="convaiinnovations/laya")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--survival", type=float, default=None)
    parser.add_argument("--pressure", type=float, default=None)
    parser.add_argument("--seconds", type=float, default=0, help="Stop after N seconds (record mode)")
    parser.add_argument("--fps", type=int, default=8, help="Demo video frame rate")
    parser.add_argument("--record", type=str, default=None, help="Write a demo video to this path")
    parser.add_argument("--poster", type=str, default=None, help="Also extract a poster frame (jpg)")
    parser.add_argument("--poster-at", type=float, default=12, help="Poster timestamp in seconds")
    parser.add_argument("--capture", type=str, default=None,
                        help="Record the app window itself to a GIF (starts once the model is loaded)")
    parser.add_argument("--capture-seconds", type=float, default=32, help="GIF length")
    args = parser.parse_args()

    if args.record:
        rec_args = argparse.Namespace(**{**vars(args), "episodes": args.episodes or 1})
        run_record(rec_args)
    else:
        run_app(args)


if __name__ == "__main__":
    main()
