from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCREEN_DIR = Path(r"C:\Users\ASUS\Desktop\screens")
OUT_DIR = ROOT / "kaggle" / "media_gallery"

W, H = 1600, 800
THUMB_W, THUMB_H = 560, 280

BG = "#071018"
PANEL = "#0d1823"
PANEL_2 = "#111f2d"
BORDER = "#29445f"
BLUE = "#8fc7ff"
BLUE_2 = "#a8c5ff"
MUTED = "#9aa8ba"
WHITE = "#f4f8ff"
GREEN = "#70e2a0"
YELLOW = "#f2c65b"
PURPLE = "#b8a5ff"


def font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts") / name,
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


FONT_DISPLAY = font("georgia.ttf", 72)
FONT_DISPLAY_SM = font("georgia.ttf", 52)
FONT_H1 = font("segoeuib.ttf", 58)
FONT_H2 = font("segoeuib.ttf", 36)
FONT_BODY = font("segoeui.ttf", 27)
FONT_BODY_SM = font("segoeui.ttf", 22)
FONT_MONO = font("consola.ttf", 21)
FONT_LABEL = font("segoeuib.ttf", 20)


@dataclass(frozen=True)
class Shot:
    slug: str
    title: str
    subtitle: str
    kicker: str
    bullets: tuple[str, ...]
    screen: str | None
    mode: str


SHOTS = [
    Shot(
        "01_cover",
        "Echo turns hidden effort into proof.",
        "A local-first Gemma 4 opportunity engine for under-observed learners.",
        "GEMMA 4 GOOD",
        ("proof from ordinary work", "practice from missing evidence", "offline continuity"),
        "Screenshot_20260514_105640.png",
        "cover",
    ),
    Shot(
        "02_problem",
        "Real ability often starts messy.",
        "Noor has repaired hardware, helped classmates, and kept rough notes. Echo treats those moments as evidence.",
        "THE PROBLEM",
        ("no mentor", "weak internet", "real work but no portfolio"),
        "Screenshot_20260514_105659.png",
        "evidence",
    ),
    Shot(
        "03_proof_capture",
        "Gemma 4 extracts skill signal.",
        "A rough prototype note becomes structured proof: skills, confidence, privacy risk, missing context, and next action.",
        "PROOF CAPTURE",
        ("sensor tested 40 min", "cost reduced $18 -> $11", "missing: feedback quote"),
        "Screenshot_20260514_105926.png",
        "proof",
    ),
    Shot(
        "04_current_read",
        "Evidence becomes a living read.",
        "Echo explains what it believes, why it believes it, and what would change the read.",
        "PATTERN MAP",
        ("repair skill", "teaching signal", "public-safe proof"),
        "Screenshot_20260514_105828.png",
        "read",
    ),
    Shot(
        "05_next_step",
        "One useful step, every day.",
        "The next move is small enough to do now and specific enough to create new proof.",
        "NEXT PROOF STEP",
        ("ask one teacher for a quote", "record one explanation", "log the outcome"),
        "Screenshot_20260514_110011.png",
        "practice",
    ),
    Shot(
        "06_opportunity",
        "Proof becomes direction.",
        "Echo maps current evidence to an opportunity plan and shows the missing proof gap.",
        "OPPORTUNITY READINESS",
        ("before: scattered effort", "after: proof card + feedback", "next: scholarship-ready story"),
        "Screenshot_20260514_105926.png",
        "opportunity",
    ),
    Shot(
        "07_runtime",
        "Local-first by design.",
        "Home Brain is strongest. This Device keeps Talk available offline. Cloud is the fallback.",
        "WHERE ECHO THINKS",
        ("Home Brain", "This Device", "Echo Cloud", "Memory Pack"),
        "Screenshot_20260514_105908.png",
        "runtime",
    ),
    Shot(
        "08_training",
        "The model improves from real signal.",
        "Saved moments, preference lessons, outcomes, and proof feed an eval-gated personal adapter path.",
        "IMPROVE ECHO",
        ("707 saved moments", "10 model updates", "bounded LoRA demo path"),
        "Screenshot_20260514_105816.png",
        "training",
    ),
]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rounded(draw: ImageDraw.ImageDraw, box, radius=28, fill=PANEL, outline=BORDER, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy, value, fill=WHITE, font_obj=FONT_BODY, anchor=None):
    draw.text(xy, value, fill=fill, font=font_obj, anchor=anchor)


def wrap(draw: ImageDraw.ImageDraw, value: str, font_obj, max_w: int) -> list[str]:
    words = value.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = f"{cur} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font_obj)[2] <= max_w:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, font_obj, fill, max_w: int, line_gap=10) -> int:
    for line in wrap(draw, value, font_obj, max_w):
        text(draw, (x, y), line, fill=fill, font_obj=font_obj)
        box = draw.textbbox((x, y), line, font=font_obj)
        y = box[3] + line_gap
    return y


def crop_phone(path: Path) -> Image.Image:
    src = Image.open(path).convert("RGB")
    # Most screenshots already include a black iPhone-shaped canvas. Preserve it.
    return src


def phone_frame(screen: Image.Image, target_h: int = 640) -> Image.Image:
    ratio = screen.width / screen.height
    target_w = int(target_h * ratio)
    img = screen.resize((target_w, target_h), Image.Resampling.LANCZOS)
    frame = Image.new("RGBA", (target_w + 34, target_h + 34), (0, 0, 0, 0))
    shadow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((12, 10, frame.width - 10, frame.height - 8), radius=72, fill=(0, 0, 0, 140))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    frame.alpha_composite(shadow)
    draw = ImageDraw.Draw(frame)
    draw.rounded_rectangle((16, 16, frame.width - 16, frame.height - 16), radius=70, fill="#05090e", outline=BORDER, width=2)
    frame.alpha_composite(img.convert("RGBA"), (17, 17))
    return frame


def draw_chip(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, fill=PANEL_2, outline=BORDER, fg=BLUE) -> int:
    box = draw.textbbox((0, 0), value, font=FONT_LABEL)
    w = box[2] - box[0] + 34
    draw.rounded_rectangle((x, y, x + w, y + 40), radius=20, fill=fill, outline=outline, width=1)
    draw.text((x + 17, y + 9), value, fill=fg, font=FONT_LABEL)
    return x + w + 12


def draw_signal_card(draw: ImageDraw.ImageDraw, x: int, y: int, title: str, value: str, color=BLUE):
    rounded(draw, (x, y, x + 255, y + 128), radius=22, fill=PANEL, outline=BORDER, width=2)
    text(draw, (x + 26, y + 25), title.upper(), fill=MUTED, font_obj=FONT_MONO)
    text(draw, (x + 26, y + 68), value, fill=color, font_obj=FONT_H2)


def gradient_bg() -> Image.Image:
    base = Image.new("RGB", (W, H), BG)
    arr = base.load()
    for y in range(H):
        for x in range(W):
            rx = x / W
            ry = y / H
            glow = max(0, 1 - ((rx - 0.78) ** 2 / 0.17 + (ry - 0.22) ** 2 / 0.15))
            glow2 = max(0, 1 - ((rx - 0.12) ** 2 / 0.11 + (ry - 0.78) ** 2 / 0.18))
            r, g, b = hex_to_rgb(BG)
            r += int(glow * 12 + glow2 * 4)
            g += int(glow * 28 + glow2 * 15)
            b += int(glow * 48 + glow2 * 26)
            arr[x, y] = (min(r, 255), min(g, 255), min(b, 255))
    return base


def compose(shot: Shot) -> Image.Image:
    img = gradient_bg().convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Left narrative panel
    text(draw, (96, 86), shot.kicker, fill=BLUE, font_obj=FONT_MONO)
    y = draw_wrapped(draw, 94, 145, shot.title, FONT_DISPLAY if len(shot.title) < 42 else FONT_DISPLAY_SM, WHITE, 700, line_gap=8)
    y += 20
    y = draw_wrapped(draw, 98, y, shot.subtitle, FONT_BODY, MUTED, 660, line_gap=10)
    y += 28
    chip_x = 98
    for bullet in shot.bullets:
        if chip_x > 600:
            chip_x = 98
            y += 52
        chip_x = draw_chip(draw, chip_x, y, bullet)

    # Small product loop strip
    strip_y = 646
    rounded(draw, (94, strip_y, 724, strip_y + 86), radius=22, fill="#0a141e", outline="#20384f", width=1)
    loop = ["Signal", "Proof", "Practice", "Opportunity"]
    x = 122
    for idx, item in enumerate(loop):
        color = [BLUE, GREEN, YELLOW, PURPLE][idx]
        text(draw, (x, strip_y + 29), item, fill=color, font_obj=FONT_LABEL)
        x += 112
        if idx < len(loop) - 1:
            text(draw, (x, strip_y + 28), "->", fill=MUTED, font_obj=FONT_MONO)
            x += 48

    # Right visual zone
    rounded(draw, (812, 74, 1506, 724), radius=40, fill="#08131d", outline="#213a54", width=2)
    if shot.screen:
        screen = crop_phone(SCREEN_DIR / shot.screen)
        frame = phone_frame(screen, target_h=612)
        img.alpha_composite(frame, (1014, 92))

    # Context cards over/around phone, tailored per shot.
    if shot.mode == "cover":
        draw_signal_card(draw, 850, 140, "Read", "strong", PURPLE)
        draw_signal_card(draw, 850, 292, "Proof", "36", BLUE_2)
        draw_signal_card(draw, 850, 444, "Missing", "0", YELLOW)
    elif shot.mode == "evidence":
        cards = [("repair", "sensor stable"), ("teaching", "peer understood"), ("constraint", "offline test")]
        cy = 134
        for title, value in cards:
            rounded(draw, (852, cy, 1118, cy + 94), radius=20, fill=PANEL, outline=BORDER, width=1)
            text(draw, (878, cy + 20), title.upper(), fill=BLUE, font_obj=FONT_MONO)
            text(draw, (878, cy + 52), value, fill=WHITE, font_obj=FONT_BODY_SM)
            cy += 116
    elif shot.mode == "proof":
        rounded(draw, (850, 514, 1198, 662), radius=24, fill=PANEL, outline=BORDER, width=2)
        text(draw, (878, 542), "GEMMA 4 OUTPUT", fill=BLUE, font_obj=FONT_MONO)
        text(draw, (878, 584), "create_proof_item", fill=GREEN, font_obj=FONT_H2)
    elif shot.mode == "read":
        rounded(draw, (852, 540, 1232, 668), radius=24, fill=PANEL, outline=BORDER, width=2)
        text(draw, (880, 566), "Evidence before insight", fill=BLUE, font_obj=FONT_H2)
        text(draw, (880, 612), "Every read cites what changed it.", fill=MUTED, font_obj=FONT_BODY_SM)
    elif shot.mode == "practice":
        rounded(draw, (852, 538, 1264, 674), radius=24, fill=PANEL, outline=BORDER, width=2)
        text(draw, (882, 566), "Today", fill=GREEN, font_obj=FONT_H2)
        text(draw, (882, 614), "Ask one reviewer. Log the result.", fill=WHITE, font_obj=FONT_BODY_SM)
    elif shot.mode == "opportunity":
        rounded(draw, (852, 522, 1220, 672), radius=24, fill=PANEL, outline=BORDER, width=2)
        text(draw, (882, 548), "Readiness", fill=BLUE, font_obj=FONT_MONO)
        text(draw, (882, 586), "32% -> 78%", fill=GREEN, font_obj=FONT_H1)
    elif shot.mode == "runtime":
        draw_signal_card(draw, 852, 128, "Home Brain", "active", BLUE)
        draw_signal_card(draw, 852, 284, "This Device", "ready", GREEN)
        draw_signal_card(draw, 852, 440, "Memory Pack", "synced", GREEN)
    elif shot.mode == "training":
        draw_signal_card(draw, 852, 130, "Saved", "707", BLUE)
        draw_signal_card(draw, 852, 286, "Lessons", "5", GREEN)
        draw_signal_card(draw, 852, 442, "Updates", "10", PURPLE)

    # Footer mark
    text(draw, (95, 756), "Echo / Gemma 4 Good Hackathon", fill="#53657a", font_obj=FONT_MONO)
    return img.convert("RGB")


def save_pair(shot: Shot):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = compose(shot)
    image.save(OUT_DIR / f"{shot.slug}.png")
    thumb = image.resize((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
    thumb.save(OUT_DIR / f"{shot.slug}_thumb_560x280.png")


def main() -> int:
    for shot in SHOTS:
        save_pair(shot)
    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
