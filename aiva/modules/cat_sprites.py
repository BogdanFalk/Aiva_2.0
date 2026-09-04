"""Sprite generator for Aiva's desktop cat.

Every frame is hand-drawn as a 16x16 grid of palette characters, rendered
with hard pixels (alpha strictly 0 or 255, no anti-aliasing) and scaled with
NEAREST so the cat stays crisp and never fringes against the overlay's
transparency key color. Regenerate any time:

    cd aiva && python -m modules.cat_sprites

Output goes to aiva/assets/cat/ (gitignored): one PNG per frame, a
manifest.json recording GENERATOR_VERSION (the runtime regenerates on
mismatch), and preview.png showing every frame for quick art review.
"""

import json
import os

from PIL import Image, ImageDraw, ImageOps

GENERATOR_VERSION = 1

LOGICAL = 16  # frames are drawn on a 16x16 grid
SCALE = 4     # rendered at 64x64

ASSET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "cat"
)
_MANIFEST = "manifest.json"

# NOTE: the overlay's transparency key is magenta #ff00fe — nothing here may
# come close to it. Soft gray cat, white points, green eyes.
PALETTE = {
    ".": None,                  # transparent
    "K": (30, 28, 34, 255),     # outline
    "G": (122, 117, 130, 255),  # body gray
    "L": (158, 152, 166, 255),  # highlight gray
    "W": (240, 238, 242, 255),  # white muzzle/paws/chest
    "P": (232, 140, 160, 255),  # pink ears/nose
    "E": (110, 210, 130, 255),  # green eyes
    "Y": (240, 196, 90, 255),   # food bowl
    "R": (232, 92, 110, 255),   # hearts / ball
    "Z": (150, 190, 235, 255),  # sleep z's
}

# Directional frames are drawn facing RIGHT; a mirrored *_left_* copy of each
# is generated automatically. Frames without "right" in the name are used
# as-is (front views / symmetric poses).
FRAMES = {
    # --- walking (side view, two-beat gait) --------------------------------
    "walk_right_0": [
        "................",
        "................",
        "...........K..K.",
        "..K........KKKK.",
        "..K.......KGGGGK",
        "...K......KGGGEK",
        "...K......KGGGGK",
        "....K..KKKGGGGWK",
        "....KKKGGGGGGGK.",
        ".....KGGGGGGGGK.",
        "......KGGGGGGK..",
        "......KGGGGGGK..",
        "......KGKKKGGK..",
        "......KGK.KGK...",
        "......KWK.KWK...",
        "................",
    ],
    "walk_right_1": [
        "................",
        "................",
        "...........K..K.",
        "..K........KKKK.",
        "..K.......KGGGGK",
        "...K......KGGGEK",
        "...K......KGGGGK",
        "....K..KKKGGGGWK",
        "....KKKGGGGGGGK.",
        ".....KGGGGGGGGK.",
        "......KGGGGGGK..",
        "......KGGGGGGK..",
        "......KGKKKGGK..",
        ".....KGK...KGK..",
        ".....KWK...KWK..",
        "................",
    ],
    # --- running / chasing (stretched gallop) ------------------------------
    "run_right_0": [
        "................",
        "................",
        "................",
        "................",
        "............K.K.",
        ".K.........KKKK.",
        "..K........KGGEK",
        "...KK..KKKKGGGGK",
        ".....KKGGGGGGGWK",
        "......KGGGGGGGK.",
        ".....KGGGGGGGK..",
        "....KGGKKKKGGK..",
        "...KGK......KGK.",
        "..KWK........KWK",
        "................",
        "................",
    ],
    "run_right_1": [
        "................",
        "................",
        "................",
        "................",
        "................",
        "............K.K.",
        ".K.........KKKK.",
        "..K........KGGEK",
        "...KK..KKKKGGGGK",
        ".....KKGGGGGGGWK",
        "......KGGGGGGGK.",
        "......KGKKKGGK..",
        "......KGK.KGK...",
        "......KWK.KWK...",
        "................",
        "................",
    ],
    # --- sitting (front view) ----------------------------------------------
    "sit": [
        "................",
        "................",
        "...K.......K....",
        "...KK.....KK....",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGEGGGGGEGK...",
        "..KGGGGPGGGGK...",
        "..KGGGWWWGGGK...",
        "...KGGGGGGGK....",
        "...KGWGGGWGK....",
        "..KGGWGGGWGGKK..",
        "..KGGGGGGGGGKGK.",
        "...KWWK.KWWK.K..",
        "................",
    ],
    "sit_blink": [
        "................",
        "................",
        "...K.......K....",
        "...KK.....KK....",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGKGGGGGKGK...",
        "..KGGGGPGGGGK...",
        "..KGGGWWWGGGK...",
        "...KGGGGGGGK....",
        "...KGWGGGWGK....",
        "..KGGWGGGWGGKK..",
        "..KGGGGGGGGGKGK.",
        "...KWWK.KWWK.K..",
        "................",
    ],
    # --- grooming (front view, paw raised to face) --------------------------
    "groom_0": [
        "................",
        "................",
        "...K.......K....",
        "...KK.....KK....",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGKGGGGGKGK...",
        "..KGGGGPGGWWK...",
        "..KGGGWWWGWWK...",
        "...KGGGGGGGK....",
        "...KGWGGGGGK....",
        "..KGGWGGGGGGKK..",
        "..KGGGGGGGGGKGK.",
        "...KWWK.KWWK.K..",
        "................",
    ],
    "groom_1": [
        "................",
        "................",
        "...K.......K....",
        "...KK.....KK....",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGKGGGGGKGK...",
        "..KGGGGPGGGGK...",
        "..KGGGWWWWWGK...",
        "...KGGGGGWWK....",
        "...KGWGGGGGK....",
        "..KGGWGGGGGGKK..",
        "..KGGGGGGGGGKGK.",
        "...KWWK.KWWK.K..",
        "................",
    ],
    # --- sleeping (curled ball, head tucked left, tail wrapped in front) -----
    "sleep_0": [
        "................",
        "................",
        "................",
        "................",
        "...K.K..........",
        "...KKKKKK.......",
        "..KGGGGGGKK.....",
        ".KGKGGGGGGGK....",
        ".KGGGGGGGGGGK...",
        ".KPGGGGGGGGGGK..",
        ".KGGGGGGGGGGGK..",
        ".KGGGGGGGGGGGK..",
        "..KLLGGGGGGGK...",
        "...KKLLLLLKK....",
        ".....KKKKK......",
        "................",
    ],
    "sleep_1": [
        "............Z...",
        "...........ZZZ..",
        "............Z...",
        "..........ZZZ...",
        "...K.K..........",
        "...KKKKKK.......",
        "..KGGGGGGKK.....",
        ".KGKGGGGGGGK....",
        ".KGGGGGGGGGGK...",
        ".KPGGGGGGGGGGK..",
        ".KGGGGGGGGGGGK..",
        ".KGGGGGGGGGGGK..",
        "..KLLGGGGGGGK...",
        "...KKLLLLLKK....",
        ".....KKKKK......",
        "................",
    ],
    # --- purring after a pet (front view + heart) ---------------------------
    "purr_0": [
        "............R.R.",
        "...........RRRRR",
        "...K........RRR.",
        "...KK.....KK.R..",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGKGGGGGKGK...",
        "..KGGGGPGGGGK...",
        "..KGGGWWWGGGK...",
        "...KGGGGGGGK....",
        "...KGWGGGWGK....",
        "..KGGWGGGWGGKK..",
        "..KGGGGGGGGGKGK.",
        "...KWWK.KWWK.K..",
        "................",
    ],
    "purr_1": [
        ".R.R............",
        "RRRRR...........",
        ".RRR.......K....",
        "..R.K.....KK....",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGKGGGGGKGK...",
        "..KGGGGPGGGGK...",
        "..KGGGWWWGGGK...",
        "...KGGGGGGGK....",
        "...KGWGGGWGK....",
        "..KGGWGGGWGGKK..",
        "..KGGGGGGGGGKGK.",
        "...KWWK.KWWK.K..",
        "................",
    ],
    # --- eating (side view at bowl) -----------------------------------------
    "eat_right_0": [
        "................",
        "................",
        "................",
        "..K.............",
        "..KK............",
        "...K............",
        "...KKKKKK.......",
        "..KGGGGGGK......",
        "..KGGGGGGGKK.K.K",
        "..KGGGGGGGGKKKK.",
        "...KGKKKGGKGGGGK",
        "...KGK.KGKKGGGK.",
        "...KWK.KWK.KGK..",
        "..........KYYYK.",
        "..........KKKKK.",
        "................",
    ],
    "eat_right_1": [
        "................",
        "................",
        "................",
        "..K.........K.K.",
        "..KK........KKKK",
        "...K.......KGGGK",
        "...KKKKKK..KGGEK",
        "..KGGGGGGKKGGGGK",
        "..KGGGGGGGGGGGK.",
        "..KGGGGGGGGGGK..",
        "...KGKKKGGKGK...",
        "...KGK.KGK.K....",
        "...KWK.KWK......",
        "..........KYYYK.",
        "..........KKKKK.",
        "................",
    ],
    # --- pounce crouch (side view, ready to spring) -------------------------
    "pounce_right_0": [
        "................",
        "................",
        "................",
        "....K...........",
        "....KK..........",
        ".....K..........",
        ".....K.......K.K",
        "....KKKKKKK..KKK",
        "...KGGGGGGGKKGGK",
        "..KGGGGGGGGGGGEK",
        "..KGGGGGGGGGGGGK",
        "..KGKKKKKGGKGGGK",
        "..KGK....KGKWWK.",
        "..KWK....KWK....",
        "................",
        "................",
    ],
    "pounce_right_1": [
        "................",
        "................",
        "................",
        "................",
        "....K...........",
        "....KK..........",
        ".....K.......K.K",
        "....KKKKKKK..KKK",
        "...KGGGGGGGKKGGK",
        "..KGGGGGGGGGGGEK",
        "..KGGGGGGGGGGGGK",
        "..KGKKKKKGGKGGGK",
        "..KGK....KGKWWK.",
        "..KWK....KWK....",
        "................",
        "................",
    ],
    # --- playing with the ball (front view, hugging then batting it) --------
    "play_0": [
        "................",
        "................",
        "...K.......K....",
        "...KK.....KK....",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGEGGGGGEGK...",
        "..KGGGGPGGGGK...",
        "..KGGGWWWGGGK...",
        "..KGWGGGGGWGK...",
        "..KGWGGRRGWGK...",
        "..KGGGGRRGGGKK..",
        "..KGGGGGGGGGKGK.",
        "...KWWK.KWWK.K..",
        "................",
    ],
    "play_1": [
        "................",
        "................",
        "...K.......K....",
        "...KK.....KK....",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGEGGGGGEGK...",
        "..KGGGGPGGGGK...",
        "..KGGGWWWGGGK...",
        "..KGGWGGGGWGK...",
        "..KGWWGGGGWGK...",
        "..KGGRRGGGGGKK..",
        "..KGGRRGGGGGKGK.",
        "...KWK..KWWK.K..",
        "................",
    ],
    # --- alert (ears up, wide eyes — reacts to being woken) ------------------
    "alert": [
        "................",
        "...K.......K....",
        "...KK.....KK....",
        "...KPK...KPK....",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGEEGGGEEGK...",
        "..KGEEGGGEEGK...",
        "..KGGGGPGGGGK...",
        "..KGGGWWWGGGK.K.",
        "...KGGGGGGGKKGK.",
        "..KGGGGGGGGGKGK.",
        "..KGGGGGGGGGKK..",
        "...KWWK.KWWK....",
        "................",
    ],
    # --- carried by the user (dangling) --------------------------------------
    "carry": [
        "................",
        "...K.......K....",
        "...KK.....KK....",
        "...KPK...KPK....",
        "...KGKKKKKGK....",
        "..KGGGGGGGGGK...",
        "..KGKGGGGGKGK...",
        "..KGGGGPGGGGK...",
        "..KGGGWWWGGGK...",
        "...KGGGGGGGK....",
        "...KGGGGGGGK....",
        "...KGGGGGGGK....",
        "...KGKGGGKGK....",
        "....KGK.KGK.....",
        "....KWK.KWK.....",
        "................",
    ],
}


def _render(rows):
    """Render one 16x16 character grid to a scaled RGBA image."""
    if len(rows) != LOGICAL:
        raise ValueError(f"frame has {len(rows)} rows, expected {LOGICAL}")
    img = Image.new("RGBA", (LOGICAL, LOGICAL), (0, 0, 0, 0))
    for y, row in enumerate(rows):
        if len(row) != LOGICAL:
            raise ValueError(f"row {y} is {len(row)} chars, expected {LOGICAL}: {row!r}")
        for x, ch in enumerate(row):
            color = PALETTE[ch]
            if color is not None:
                img.putpixel((x, y), color)
    return img.resize((LOGICAL * SCALE, LOGICAL * SCALE), Image.NEAREST)


def _write_preview(images, out_dir):
    """One sheet with every frame labelled, for eyeballing the art."""
    from PIL import ImageFont

    cell_w, cell_h, cols = 116, 130, 6
    names = sorted(images)
    rows = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (210, 210, 214))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, name in enumerate(names):
        cx, cy = (i % cols) * cell_w, (i // cols) * cell_h
        # checkerboard behind each frame so transparency reads
        for ty in range(0, 96, 12):
            for tx in range(0, 96, 12):
                if (tx // 12 + ty // 12) % 2 == 0:
                    draw.rectangle(
                        [cx + 10 + tx, cy + 6 + ty, cx + 10 + tx + 11, cy + 6 + ty + 11],
                        fill=(184, 184, 190),
                    )
        frame = images[name].resize((96, 96), Image.NEAREST)
        sheet.paste(frame, (cx + 10, cy + 6), frame)
        draw.text((cx + 10, cy + 106), name, fill=(40, 40, 44), font=font)
    sheet.save(os.path.join(out_dir, "preview.png"))


def generate(out_dir=ASSET_DIR):
    """Render every frame (plus mirrored left variants) into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    images = {}
    for name, rows in FRAMES.items():
        img = _render(rows)
        images[name] = img
        if "right" in name:
            images[name.replace("right", "left")] = ImageOps.mirror(img)
    for name, img in images.items():
        img.save(os.path.join(out_dir, f"{name}.png"))
    _write_preview(images, out_dir)
    with open(os.path.join(out_dir, _MANIFEST), "w") as f:
        json.dump({"version": GENERATOR_VERSION}, f)
    return sorted(images)


def ensure_sprites(out_dir=ASSET_DIR):
    """Regenerate the sprite PNGs if missing or from an older generator."""
    try:
        with open(os.path.join(out_dir, _MANIFEST)) as f:
            if json.load(f).get("version") == GENERATOR_VERSION:
                return False
    except (OSError, ValueError):
        pass
    generate(out_dir)
    return True


if __name__ == "__main__":
    names = generate()
    print(f"Rendered {len(names)} frames into {ASSET_DIR}")
    print(f"Review: {os.path.join(ASSET_DIR, 'preview.png')}")
