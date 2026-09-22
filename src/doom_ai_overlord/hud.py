"""HUD overlay burned into recorded video frames, plus hardware info."""

import torch
from PIL import Image, ImageDraw, ImageFont


def _pil_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bar_color(frac):
    if frac > 0.5:
        return (90, 200, 90)
    if frac > 0.25:
        return (230, 200, 60)
    return (230, 80, 70)


def hud_overlay(frame_rgb, ev, hw_line, flashes):
    """Compose the HUD on a 640x480 RGB frame."""
    img = Image.fromarray(frame_rgb).convert("RGBA")
    w, h = img.size
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f_big, f_sm, f_tiny = _pil_font(18), _pil_font(14), _pil_font(11)

    d.rectangle([0, 0, w, 34], fill=(0, 0, 0, 165))
    hp = ev["health"]
    d.text((8, 1), "HP", font=f_tiny, fill=(200, 200, 200, 255))
    if hp is not None:
        frac = max(0.0, min(1.0, hp / 100.0))
        d.rectangle([8, 15, 150, 29], fill=(70, 70, 70, 255))
        d.rectangle([8, 15, 8 + int(142 * frac), 29], fill=_bar_color(frac))
        d.text((156, 8), str(hp), font=f_big, fill=(255, 255, 255, 255))
    else:
        d.text((8, 15), "n/a", font=f_sm, fill=(200, 200, 200, 255))
    ammo = ev["geom"]["ammo"]
    d.text((220, 1), "AMMO", font=f_tiny, fill=(200, 200, 200, 255))
    if ammo is not None:
        frac = max(0.0, min(1.0, ammo / 50.0))
        d.rectangle([220, 15, 320, 29], fill=(70, 70, 70, 255))
        d.rectangle([220, 15, 220 + int(100 * frac), 29], fill=_bar_color(frac))
        d.text((326, 8), str(ammo), font=f_big, fill=(255, 255, 255, 255))
    else:
        d.text((220, 15), "n/a", font=f_sm, fill=(200, 200, 200, 255))
    d.text((400, 8), f"FRAGS {ev['kills']}", font=f_big, fill=(255, 220, 120, 255))
    d.text((520, 10), f"{ev['episode_tics']} tics", font=f_sm, fill=(220, 220, 220, 255))

    d.rectangle([0, 34, w, 52], fill=(0, 0, 0, 120))
    d.text((8, 37), hw_line, font=f_tiny, fill=(170, 170, 170, 255))

    d.rectangle([0, h - 42, w, h], fill=(0, 0, 0, 165))
    src_col = (120, 220, 130, 255) if not ev["used_fallback"] else (240, 170, 80, 255)
    d.text((10, h - 34), f"ACT {ev['choice']}", font=f_sm, fill=src_col)
    d.text((10, h - 16), f"{ev['conf'] * 100:3.0f}% conf · {ev['dt_ms']:3.0f} ms · {ev['tics']} tics",
           font=f_tiny, fill=(200, 200, 200, 255))
    d.text((250, h - 34), f"PRIO {ev['priority']} ({ev['priority_src']})", font=f_sm,
           fill=(160, 200, 255, 255))
    dp = ev["danger_p"]
    d.rectangle([430, h - 30, 540, h - 16], fill=(70, 70, 70, 255))
    d.rectangle([430, h - 30, 430 + int(110 * dp), h - 16], fill=_bar_color(1.0 - dp))
    d.text((546, h - 34), f"danger {dp:.2f}", font=f_tiny, fill=(220, 220, 220, 255))
    tgt = ev["state_dict"]
    d.text((430, h - 16), f"{tgt['target']}/{tgt['range']} {ev['geom']['offset_px']}px",
           font=f_tiny, fill=(200, 200, 200, 255))

    y = 60
    for text_s, color, _expiry in flashes:
        box = d.textbbox((0, 0), text_s, font=f_big)
        x = (w - (box[2] - box[0])) // 2
        d.text((x, y), text_s, font=f_big, fill=color, stroke_width=2, stroke_fill=(0, 0, 0, 255))
        y += 26

    return Image.alpha_composite(img, ov).convert("RGB")


def hardware_info(args, scenario_base, goals):
    cuda = torch.cuda.is_available()
    return {
        "GPU": torch.cuda.get_device_name(0) if cuda else "CPU only",
        "torch": torch.__version__,
        "CUDA": torch.version.cuda or "-",
        "model": args.model,
        "device": args.device,
        "scenario": scenario_base,
        "goals": f"survival={goals['survival']}, pressure={goals['pressure']}",
    }
