"""Procedural leaf images for smoke-testing the pipeline without a dataset.

Real training uses PlantVillage / IP102 / district scouting photos. Those are
gigabytes and cannot be vendored, which makes a repository that "only works if
you first download 5 GB" impossible to review or CI.

So we synthesise. Every class is rendered with a visual signature derived from
its own ``symptoms`` text in the taxonomy - a class described with "concentric
rings" gets ringed lesions, one described with "pustules" gets pustules, one
described with "interveinal" gets green veins on a yellow lamina. The images
are cartoonish, but they are *separable in the way the real classes are
separable*, so a training run on them exercises exactly the code paths a real
run does and converges to a meaningful accuracy rather than to chance.

Nothing here is a substitute for field data. It is a substitute for waiting.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..taxonomy import CropClass, Taxonomy, load_taxonomy

# Lesion vocabulary. Order matters only for reproducibility of the rule scan.
LESION_STYLES = (
    "none",
    "ring_spot",       # concentric target lesions (Alternaria, target spot)
    "water_soaked",    # spreading dark blotch with pale halo (late blight)
    "small_spot",      # many small dark spots (septoria, brown spot, bacterial)
    "pustule",         # raised orange/yellow pustules (rusts)
    "powder",          # white/grey powdery film (powdery mildew)
    "downy",           # pale upper blotch, white sporulation (downy mildew)
    "stripe",          # elongated lesions along the veins (blast, NLB, GLS)
    "mottle",          # light/dark green mosaic (mosaic viruses)
    "curl",            # deformed cupped lamina (leaf curl viruses)
    "insect",          # visible bodies on the lamina (aphid, whitefly, hopper)
    "chew",            # ragged holes / windowpane feeding (armyworm, borer)
    "mine",            # serpentine or blotch mines (leaf miner)
    "stipple",         # fine pale stippling + webbing (mites, thrips)
    "interveinal",     # yellow lamina, green veins (Fe, Zn, Mg)
    "margin_scorch",   # necrotic leaf margins (K deficiency, heat)
    "uniform_chlorosis",  # whole-leaf yellowing (N deficiency, waterlogging)
    "purple_tint",     # anthocyanin flush (P deficiency)
    "roll",            # inward rolled, dull grey-green (water stress)
    "bleach",          # bleached papery patches (sunscald)
)

# Keyword -> style, scanned per category. Gating by category matters: an aphid
# description mentions "curling" and a potassium description mentions "yellow",
# but an aphid image must show insects and a potassium image must show a scorched
# margin. Only styles that are plausible for the category are ever considered.
_STYLE_RULES: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "disease": (
        (("concentric", "target pattern", "ringed", "rings"), "ring_spot"),
        (("water-soaked", "water soaked"), "water_soaked"),
        (("pustule",), "pustule"),
        (("powdery",), "powder"),
        (("downy", "sporulation", "velvety"), "downy"),
        (("mottling", "mosaic"), "mottle"),
        (("curling", "cupping", "crinkling"), "curl"),
        (("stripe", "streak", "cigar", "spindle", "rectangular", "elongated",
          "long grey-green", "eye-shaped", "tiger-stripe"), "stripe"),
        (("interveinal",), "interveinal"),
        (("yellowing", "yellow", "chlorosis", "orange-yellow"), "uniform_chlorosis"),
        (("spots", "blotch", "lesion", "angular"), "small_spot"),
    ),
    "pest": (
        (("mine", "mined"), "mine"),
        (("stippling", "webbing", "silvery streaks", "silvery"), "stipple"),
        (("windowpane", "ragged", "bore hole", "dead heart", "folded",
          "scraped", "mines", "holes"), "chew"),
        (("colonies", "adults", "nymphs", "hoppers", "waxy cottony", "larva",
          "moths", "hopper"), "insect"),
    ),
    "deficiency": (
        (("interveinal",), "interveinal"),
        (("purple", "reddish tints"), "purple_tint"),
        (("margins", "marginal", "scorching"), "margin_scorch"),
        (("uniform pale", "yellow", "chlorosis"), "uniform_chlorosis"),
    ),
    "abiotic": (
        (("rolling", "rolled", "wilting"), "roll"),
        (("bleached", "papery", "sunscald"), "bleach"),
        (("margins", "marginal", "necrotic"), "margin_scorch"),
        (("yellowing", "yellow"), "uniform_chlorosis"),
    ),
}

#: Used when no keyword in a category's rule list matches.
_FALLBACK_STYLE = {
    "disease": "small_spot",
    "pest": "insect",
    "deficiency": "uniform_chlorosis",
    "abiotic": "uniform_chlorosis",
}

_LESION_COLOURS: dict[str, tuple[int, int, int]] = {
    "ring_spot": (96, 62, 32),
    "water_soaked": (54, 48, 40),
    "small_spot": (110, 74, 40),
    "pustule": (206, 126, 32),
    "powder": (228, 228, 220),
    "downy": (232, 236, 220),
    "stripe": (150, 126, 70),
    "mottle": (176, 206, 110),
    "curl": (168, 186, 92),
    "insect": (58, 52, 46),
    "chew": (74, 62, 48),
    "mine": (206, 200, 150),
    "stipple": (208, 206, 176),
    "interveinal": (222, 214, 96),
    "margin_scorch": (150, 106, 52),
    "uniform_chlorosis": (214, 208, 104),
    "purple_tint": (122, 82, 138),
    "roll": (126, 140, 96),
    "bleach": (238, 236, 216),
    "none": (60, 120, 50),
}


@dataclass(frozen=True)
class LeafStyle:
    """Deterministic rendering recipe for one class."""

    class_id: str
    style: str
    leaf_rgb: tuple[int, int, int]
    lesion_rgb: tuple[int, int, int]
    density: float          # lesions per image (or intensity for diffuse styles)
    size: float             # lesion radius as a fraction of image width
    seed: int


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "big")


def style_for_class(crop_class: CropClass) -> LeafStyle:
    """Derive a rendering recipe from the class's own agronomic description."""
    haystack = f"{crop_class.symptoms} {crop_class.name}".lower()
    rules = _STYLE_RULES.get(crop_class.category)
    style = "none"
    if rules is not None:
        style = _FALLBACK_STYLE[crop_class.category]
        for keywords, candidate in rules:
            if any(k in haystack for k in keywords):
                style = candidate
                break

    seed = _stable_seed(crop_class.id)
    rng = random.Random(seed)

    # Crop identity shows up as a base leaf hue, so a maize class and a tomato
    # class differ even when they share a lesion style.
    crop_rng = random.Random(_stable_seed(crop_class.crop))
    leaf_rgb = (
        crop_rng.randint(38, 78),
        crop_rng.randint(96, 148),
        crop_rng.randint(34, 70),
    )
    base = _LESION_COLOURS[style]
    lesion_rgb = tuple(  # small per-class jitter keeps sibling classes distinct
        max(0, min(255, c + rng.randint(-18, 18))) for c in base
    )
    return LeafStyle(
        class_id=crop_class.id,
        style=style,
        leaf_rgb=leaf_rgb,
        lesion_rgb=lesion_rgb,  # type: ignore[arg-type]
        density=rng.uniform(0.55, 1.0),
        size=rng.uniform(0.030, 0.065),
        seed=seed,
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _soil_background(size: int, rng: random.Random) -> Image.Image:
    """Blurred earth-tone noise, so the model cannot key on a flat background."""
    small = 16
    arr = np.zeros((small, small, 3), dtype=np.uint8)
    base = np.array([rng.randint(84, 140), rng.randint(64, 108), rng.randint(46, 84)])
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 18, (small, small, 3))
    arr[:] = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(radius=size / 40))


def _leaf_polygon(size: int, rng: random.Random, roll: float = 0.0) -> list[tuple[float, float]]:
    """Pointed-ellipse leaf outline; ``roll`` narrows it to mimic leaf rolling."""
    cx, cy = size / 2, size / 2
    rx = size * rng.uniform(0.30, 0.38) * (1.0 - 0.55 * roll)
    ry = size * rng.uniform(0.40, 0.46)
    rot = math.radians(rng.uniform(-28, 28))
    pts = []
    for i in range(72):
        t = 2 * math.pi * i / 72
        # |sin| exponent pinches both ends into a leaf tip
        r_scale = abs(math.sin(t)) ** 0.55
        x = rx * math.cos(t)
        y = ry * math.sin(t) * (0.55 + 0.45 * r_scale)
        x += rng.uniform(-1.5, 1.5)
        pts.append(
            (cx + x * math.cos(rot) - y * math.sin(rot), cy + x * math.sin(rot) + y * math.cos(rot))
        )
    return pts


def _draw_veins(draw: ImageDraw.ImageDraw, size: int, rng: random.Random, colour, width: int = 2):
    cx, cy = size / 2, size / 2
    draw.line([(cx, cy - size * 0.42), (cx, cy + size * 0.42)], fill=colour, width=width + 1)
    for i in range(-4, 5):
        if i == 0:
            continue
        y = cy + i * size * 0.075
        spread = size * 0.26 * (1 - abs(i) / 6.0)
        draw.line([(cx, y), (cx - spread, y + size * 0.05)], fill=colour, width=width)
        draw.line([(cx, y), (cx + spread, y + size * 0.05)], fill=colour, width=width)


def _apply_style(
    img: Image.Image,
    mask: Image.Image,
    style: LeafStyle,
    size: int,
    rng: random.Random,
    background: Image.Image | None = None,
) -> Image.Image:
    """Paint the lesion signature, confined to the leaf via ``mask``."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    holes = Image.new("L", (size, size), 0)  # chewed-through tissue
    col = style.lesion_rgb
    n = max(1, int(style.density * 26))
    r = style.size * size

    def rand_pt(margin=0.30):
        return (
            rng.uniform(size * margin, size * (1 - margin)),
            rng.uniform(size * 0.14, size * 0.86),
        )

    s = style.style
    if s == "ring_spot":
        for _ in range(max(3, n // 4)):
            x, y = rand_pt()
            rr = r * rng.uniform(1.4, 2.4)
            d.ellipse([x - rr * 1.25, y - rr * 1.25, x + rr * 1.25, y + rr * 1.25],
                      fill=(*_lighten(col, 60), 110))          # chlorotic halo
            for k, frac in enumerate((1.0, 0.72, 0.46, 0.24)):  # concentric rings
                shade = _lighten(col, 34 if k % 2 else 0)
                d.ellipse([x - rr * frac, y - rr * frac, x + rr * frac, y + rr * frac],
                          fill=(*shade, 235))
    elif s == "water_soaked":
        for _ in range(max(2, n // 8)):
            x, y = rand_pt(0.22)
            rr = r * rng.uniform(3.0, 5.0)
            d.ellipse([x - rr * 1.2, y - rr, x + rr * 1.2, y + rr],
                      fill=(*_lighten(col, 95), 120))
            d.ellipse([x - rr * 0.85, y - rr * 0.72, x + rr * 0.85, y + rr * 0.72],
                      fill=(*col, 225))
    elif s == "small_spot":
        for _ in range(n * 2):
            x, y = rand_pt(0.24)
            rr = r * rng.uniform(0.3, 0.6)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(*col, 230))
            d.ellipse([x - rr * 0.45, y - rr * 0.45, x + rr * 0.45, y + rr * 0.45],
                      fill=(*_lighten(col, 70), 220))
    elif s == "pustule":
        for _ in range(n * 3):
            x, y = rand_pt(0.24)
            rr = r * rng.uniform(0.18, 0.33)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(*col, 245))
            d.ellipse([x - rr * 0.4, y - rr * 0.4, x + rr * 0.2, y + rr * 0.2],
                      fill=(*_lighten(col, 55), 200))
    elif s in ("powder", "downy", "stipple", "bleach"):
        alpha = {"powder": 150, "downy": 140, "stipple": 130, "bleach": 175}[s]
        for _ in range(n * 3):
            x, y = rand_pt(0.22)
            rr = r * rng.uniform(0.8, 2.0)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(*col, alpha))
        layer = layer.filter(ImageFilter.GaussianBlur(radius=size / 90))
        if s == "stipple":  # webbing strands
            d2 = ImageDraw.Draw(layer)
            for _ in range(10):
                p1, p2 = rand_pt(0.22), rand_pt(0.22)
                d2.line([p1, p2], fill=(*_lighten(col, 40), 190), width=1)
    elif s == "stripe":
        for _ in range(max(3, n // 3)):
            x, y = rand_pt(0.26)
            length = size * rng.uniform(0.14, 0.30)
            width = r * rng.uniform(0.45, 0.9)
            ang = math.radians(rng.uniform(70, 110))
            dx, dy = math.cos(ang) * length / 2, math.sin(ang) * length / 2
            d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=(*col, 235), width=int(max(2, width)))
            d.line([(x - dx * 0.6, y - dy * 0.6), (x + dx * 0.6, y + dy * 0.6)],
                   fill=(*_lighten(col, 60), 220), width=int(max(1, width * 0.45)))
    elif s == "mottle":
        for _ in range(n * 2):
            x, y = rand_pt(0.20)
            rr = r * rng.uniform(1.2, 3.0)
            shade = _lighten(col, rng.randint(-40, 60))
            d.ellipse([x - rr * 1.3, y - rr, x + rr * 1.3, y + rr], fill=(*shade, 140))
        layer = layer.filter(ImageFilter.GaussianBlur(radius=size / 70))
    elif s == "insect":
        for _ in range(n * 3):
            x, y = rand_pt(0.24)
            rr = r * rng.uniform(0.12, 0.26)
            d.ellipse([x - rr * 1.5, y - rr, x + rr * 1.5, y + rr], fill=(*col, 250))
            d.point((x, y - rr), fill=(*_lighten(col, 90), 255))
    elif s == "chew":
        hd = ImageDraw.Draw(holes)
        for _ in range(max(3, n // 4)):
            x, y = rand_pt(0.24)
            rr = r * rng.uniform(1.4, 2.8)
            # necrotic rim around the bite, then the bite itself punched through
            d.ellipse([x - rr * 1.35, y - rr * 1.1, x + rr * 1.35, y + rr * 1.1],
                      fill=(*col, 255))
            hd.ellipse([x - rr, y - rr * 0.8, x + rr, y + rr * 0.8], fill=255)
        for _ in range(n * 2):  # windowpane scarring: epidermis left intact
            x, y = rand_pt(0.20)
            rr = r * rng.uniform(0.7, 1.6)
            d.ellipse([x - rr, y - rr * 0.7, x + rr, y + rr * 0.7],
                      fill=(*_lighten(col, 120), 165))
    elif s == "mine":
        for _ in range(max(2, n // 6)):
            x, y = rand_pt(0.26)
            pts = [(x, y)]
            for _ in range(14):  # serpentine walk
                x += rng.uniform(-r, r) * 1.3
                y += rng.uniform(-r * 0.7, r * 0.7)
                pts.append((x, y))
            d.line(pts, fill=(*col, 235), width=int(max(2, r * 0.55)), joint="curve")
    elif s == "interveinal":
        d.rectangle([0, 0, size, size], fill=(*col, 190))
        _draw_veins(d, size, rng, (*_darken(style.leaf_rgb, 10), 255), width=3)
    elif s == "uniform_chlorosis":
        d.rectangle([0, 0, size, size], fill=(*col, 165))
    elif s == "purple_tint":
        d.rectangle([0, 0, size, size], fill=(*col, 105))
        _draw_veins(d, size, rng, (*col, 220), width=3)
    elif s == "margin_scorch":
        band = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        bd = ImageDraw.Draw(band)
        bd.rectangle([0, 0, size, size], fill=(*col, 235))
        inner = mask.filter(ImageFilter.MinFilter(9)).filter(ImageFilter.MinFilter(9))
        band.putalpha(Image.eval(inner, lambda v: 255 - v))
        layer = Image.alpha_composite(layer, band)
    elif s in ("curl", "roll"):
        d.rectangle([0, 0, size, size], fill=(*col, 120))

    layer.putalpha(_intersect_alpha(layer, mask))
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    if background is not None and holes.getbbox() is not None:
        # Chewed tissue is gone: the soil behind the canopy shows through.
        out = Image.composite(background.convert("RGB"), out, _and_masks(holes, mask))
    return out


def _and_masks(a: Image.Image, b: Image.Image) -> Image.Image:
    return Image.fromarray(
        (np.asarray(a, dtype=np.uint16) * np.asarray(b, dtype=np.uint16) // 255).astype(np.uint8)
    )


def _intersect_alpha(layer: Image.Image, mask: Image.Image) -> Image.Image:
    a = layer.getchannel("A")
    return Image.fromarray(
        (np.asarray(a, dtype=np.uint16) * np.asarray(mask, dtype=np.uint16) // 255).astype(np.uint8)
    )


def _lighten(c: Sequence[int], amount: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(v) + amount)) for v in c)  # type: ignore[return-value]


def _darken(c: Sequence[int], amount: int) -> tuple[int, int, int]:
    return _lighten(c, -amount)


def render_background(size: int, seed: int) -> Image.Image:
    """A frame with no diagnosable leaf: soil, sky, hand, machinery.

    The ``background`` class is not filler. Without non-leaf images in training,
    a classifier assigns every photo to some disease with high confidence -
    including a photo of the sky - and the advisory layer has no way to know.
    """
    rng = random.Random(seed)
    kind = rng.choice(("soil", "sky", "hand", "clutter"))
    if kind == "sky":
        arr = np.zeros((size, size, 3), dtype=np.float32)
        top = np.array([120 + rng.randint(-20, 20), 160, 225], dtype=np.float32)
        bottom = np.array([200, 215, 235], dtype=np.float32)
        for y in range(size):
            arr[y] = top + (bottom - top) * (y / size)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        for _ in range(rng.randint(0, 3)):  # clouds
            d = ImageDraw.Draw(img)
            cx, cy = rng.uniform(0, size), rng.uniform(0, size * 0.6)
            r = rng.uniform(size * 0.08, size * 0.25)
            d.ellipse([cx - r, cy - r * 0.5, cx + r, cy + r * 0.5], fill=(245, 246, 248))
        img = img.filter(ImageFilter.GaussianBlur(radius=size / 30))
    elif kind == "hand":
        img = _soil_background(size, rng)
        d = ImageDraw.Draw(img)
        skin = (rng.randint(140, 205), rng.randint(100, 150), rng.randint(80, 120))
        cx = rng.uniform(size * 0.2, size * 0.8)
        d.ellipse([cx - size * 0.3, size * 0.35, cx + size * 0.3, size * 1.3], fill=skin)
        for i in range(4):  # fingers
            fx = cx - size * 0.24 + i * size * 0.16
            d.ellipse([fx, size * 0.12, fx + size * 0.12, size * 0.55], fill=skin)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    elif kind == "clutter":
        img = _soil_background(size, rng)
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(6, 14)):  # dry stubble / straw
            x0, y0 = rng.uniform(0, size), rng.uniform(0, size)
            d.line(
                [(x0, y0), (x0 + rng.uniform(-size / 3, size / 3), y0 + rng.uniform(-size / 3, size / 3))],
                fill=(rng.randint(150, 205), rng.randint(130, 180), rng.randint(80, 125)),
                width=rng.randint(2, 6),
            )
    else:
        img = _soil_background(size, rng)
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(8, 20)):  # stones and clods
            x, y = rng.uniform(0, size), rng.uniform(0, size)
            r = rng.uniform(size * 0.02, size * 0.09)
            g = rng.randint(90, 165)
            d.ellipse([x - r, y - r * 0.8, x + r, y + r * 0.8], fill=(g, int(g * 0.85), int(g * 0.7)))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    arr = np.asarray(img).astype(np.float32) * rng.uniform(0.8, 1.2)
    arr += np.random.default_rng(rng.randrange(2**32)).normal(0, 5.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_leaf(
    style: LeafStyle,
    size: int = 224,
    seed: int | None = None,
    severity: str = "moderate",
) -> Image.Image:
    """Render one synthetic field photo for ``style``.

    ``severity`` scales how much of the leaf the signature covers, which gives
    the severity head something real to learn on synthetic data.
    """
    scale = {"none": 0.0, "low": 0.40, "moderate": 1.0, "severe": 1.9}.get(severity, 1.0)
    if scale != 1.0 and style.style != "none":
        style = LeafStyle(
            class_id=style.class_id,
            style=style.style,
            leaf_rgb=style.leaf_rgb,
            lesion_rgb=style.lesion_rgb,
            density=style.density * scale,
            size=style.size * (0.65 + 0.35 * scale),
            seed=style.seed,
        )
    rng = random.Random(style.seed if seed is None else seed)
    bg = _soil_background(size, rng)
    img = bg

    roll = 1.0 if style.style in ("roll", "curl") else 0.0
    poly = _leaf_polygon(size, rng, roll=roll)

    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))

    leaf = Image.new("RGB", (size, size), style.leaf_rgb)
    ld = ImageDraw.Draw(leaf)
    for i in range(size):  # top-lit shading gradient across the lamina
        shade = int(26 * math.cos(math.pi * i / size))
        ld.line([(0, i), (size, i)], fill=_lighten(style.leaf_rgb, shade))
    _draw_veins(ld, size, rng, _lighten(style.leaf_rgb, 42))

    img = Image.composite(leaf, img, mask)
    if style.style != "none" and style.density > 0:
        img = _apply_style(img, mask, style, size, rng, background=bg)

    # Camera realism: slight defocus, sensor noise, exposure swing.
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.9)))
    arr = np.asarray(img).astype(np.float32)
    arr *= rng.uniform(0.80, 1.20)
    arr += np.random.default_rng(rng.randrange(2**32)).normal(0, 5.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate_dataset(
    out_dir: str | Path,
    taxonomy: Taxonomy | None = None,
    class_ids: Sequence[str] | None = None,
    per_class: int = 24,
    size: int = 224,
    seed: int = 0,
    imbalance: bool = False,
) -> Path:
    """Write an ImageFolder-style synthetic dataset.

    ``imbalance=True`` makes rare classes genuinely rare (a 6:1 spread), which
    is how real scouting archives look and which exercises the class-balancing
    code in the training loop.
    """
    tax = taxonomy or load_taxonomy()
    ids = list(class_ids) if class_ids else list(tax.class_ids)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    severities = ("low", "moderate", "severe")
    for ci, class_id in enumerate(ids):
        crop_class = tax[class_id]
        style = style_for_class(crop_class)
        count = per_class
        if imbalance:
            count = max(4, int(per_class * (1.0 if ci % 4 else 0.18)))
        cdir = root / class_id
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img_seed = _stable_seed(f"{class_id}:{seed}:{i}")
            sev = "none" if not crop_class.is_actionable else severities[i % len(severities)]
            img = (
                render_background(size, img_seed)
                if crop_class.category == "background"
                else render_leaf(style, size=size, seed=img_seed, severity=sev)
            )
            # one rendered leaf == one independent sample == one split group
            name = f"{class_id}__leaf{i:04d}__sev-{sev}.jpg"
            img.save(cdir / name, quality=rng.randint(72, 94))
    return root


__all__ = [
    "LESION_STYLES",
    "LeafStyle",
    "style_for_class",
    "render_leaf",
    "render_background",
    "generate_dataset",
]
