#!/usr/bin/env python3
"""
colorcode.py
Python port of picturecolors/index.php: dominant colors of each image, written as one
image per input: the photo on top, below it a color strip (segment width = share of all
pixels, unlisted rest in light gray) and a legend with swatch, name, hex and share.
Names German by default (-lang en for English, -lang both for "German => English" as in the PHP).
Below the legend: text recommendation for the DxO PhotoLab 10 Color-Grading tool
(Farbton, Sättigung, Luminanz for Allgemein, Schatten, Mitteltöne, Spitzlichter), derived from the mean
color cast (a*, b*) of each tonal range. DxO's internal scaling is not documented;
the values are start values to be judged on the image.

Behaviour follows index.php + GetMostCommonColors (colors.inc.php):
  preview 150 px wide, per-channel rounding to multiples of delta, optional merge of
  near colors (reduce_gradients), top N colors, nearest name by squared RGB distance,
  a color whose German name was already shown is skipped.
colors.inc.php was not available; the quantization/gradient merge is a re-implementation.

Deliberate difference from the PHP: in colors.names.php only $coloren is array_flip()ped
(name => hex); $colorde stays hex => name. index.php nevertheless runs hexdiff() on the
VALUES of $colorde, i.e. on German names, so the German lookup in the PHP compared
hex digits parsed out of words and was not a nearest-color match. Here both tables are
read as hex => name, so the German name is a real nearest match.

USAGE
  Directory, PHP-equivalent settings (15 colors, delta 16, gradients merged, RGB distance):
    python colorcode.py -in cache/ -names colors.names.php

  PNG output, 2400 px wide, 3 legend columns, own font:
    python colorcode.py -in cache/ -names colors.names.php -format png -width 2400 -cols 3 -font /path/Font.ttf

  DxO Color-Grading recommendation (text below the legend: Allgemein, Schatten,
  Mitteltöne, Spitzlichter, Balance): reinforce existing mood (default), neutralize casts,
  complementary split, or none:
    python colorcode.py -in cache/ -names colors.names.php -grading reinforce
    python colorcode.py -in cache/ -names colors.names.php -grading neutral
    python colorcode.py -in cache/ -names colors.names.php -grading split
    python colorcode.py -in cache/ -names colors.names.php -grading off

  Stronger or weaker Sättigung values (Sättigung = cgmin + cgscale x cast C*, max cgmax):
    python colorcode.py -in cache/ -names colors.names.php -cgscale 5 -cgmin 20 -cgmax 80

  Calibrate: compare <stem>_grading_preview.jpg with DxO at the recommended values and set
  -cglab (preview strength) and -cgscale until both match:
    python colorcode.py -in photo.jpg -names colors.names.php -cglab 0.4

  English labels, or both as in the PHP:
    python colorcode.py -in cache/ -names colors.names.php -lang en
    python colorcode.py -in cache/ -names colors.names.php -lang both

  Single image, 10 colors, coarser rounding:
    python colorcode.py -in photo.jpg -names colors.names.php -ncolors 10 -delta 24

  PHP reduce_gradients=false:
    python colorcode.py -in cache/ -names colors.names.php -keepgradients

  Perceptual name matching (CIEDE2000) and merged instead of skipped duplicates:
    python colorcode.py -in cache/ -names colors.names.php -dist de2000 -dedup merge

  k-means in CIELAB instead of rounding:
    python colorcode.py -in cache/ -names colors.names.php -mode kmeans -k 8

REQUIRED INPUTS
  -in     image file or directory (jpg, jpeg, png, tif, tiff, webp, bmp); directory filter as
          in index.php: skips dotfiles, "liquid*", index.html
  -names  colors.names.php (both arrays "HEX" => "Name") or JSON {"de": {name: hex}, "en": {...}}

OUTPUTS (-out, default colorcode_out)
  <stem>_colors.jpg   photo + color strip + legend (German names by default)
                      + DxO Color-Grading as text: Farbton / Sättigung / Luminanz start values
                        for Allgemein, Schatten, Mitteltöne, Spitzlichter, and Balance
  <stem>.json         also contains dxo_color_grading incl. measured casts per tonal range
  <stem>_grading_preview.jpg   approximate look of the recommendation, single image (-nopreview to skip)
  <stem>.json   hex, share, DE and EN name with distance per color (always both)
  summary.csv   one row per image and color

DEPENDENCIES
  Python >= 3.9, numpy, Pillow (>= 10.1 for the built-in scalable fallback font)
  Font: DejaVu Sans or Arial are used if found (umlauts), otherwise Pillow's default font
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

# ===== color math (Lab only needed for -dist de2000 and -mode kmeans)

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    c = np.asarray(rgb, float) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    e, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > e, np.cbrt(xyz), (k * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    e, k = 216 / 24389, 24389 / 27
    xyz = np.where(f ** 3 > e, f ** 3, (116 * f - 16) / k)
    xyz[..., 1] = np.where(lab[..., 0] > k * e, fy ** 3, lab[..., 0] / k)
    lin = np.clip((xyz * _WHITE) @ np.linalg.inv(_M).T, 0, 1)
    return np.clip(np.rint(np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255), 0, 255)


def delta_e2000(lab1, lab2):
    L1, a1, b1 = (lab1[:, None, i] for i in range(3))
    L2, a2, b2 = (lab2[None, :, i] for i in range(3))
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360
    h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    z = (C1p * C2p) == 0
    dh = h2 - h1
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(z, 0, dh)
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lb, Cb = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1 + h2
    hb = np.where(np.abs(h1 - h2) > 180, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2), hs / 2)
    hb = np.where(z, hs, hb)
    r = np.radians
    T = (1 - 0.17 * np.cos(r(hb - 30)) + 0.24 * np.cos(r(2 * hb))
         + 0.32 * np.cos(r(3 * hb + 6)) - 0.20 * np.cos(r(4 * hb - 63)))
    Rc = 2 * np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7))
    Rt = -np.sin(r(60 * np.exp(-((hb - 275) / 25) ** 2))) * Rc
    Sl = 1 + 0.015 * (Lb - 50) ** 2 / np.sqrt(20 + (Lb - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cb, 1 + 0.015 * Cb * T
    dL, dC = L2 - L1, C2p - C1p
    return np.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2 + Rt * (dC / Sc) * (dH / Sh))


# ===== names

def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def load_names(path):
    p = Path(path)
    if not p.is_file():
        sys.exit(f"names file not found: {path}")
    txt = p.read_text(encoding="utf-8", errors="replace")
    raw = {}
    if p.suffix.lower() == ".php":
        for var, lang in (("colorde", "de"), ("coloren", "en")):
            m = re.search(r"\$" + var + r"\s*=\s*array\((.*?)\);", txt, re.S)
            if not m:
                sys.exit(f"${var} = array(...) not found in {path}")
            d = {}
            for hx, nm in re.findall(r'"([0-9A-Fa-f]{6})"\s*=>\s*"([^"]*)"', m.group(1)):
                d[hx.upper()] = nm.strip()  # duplicate keys: last wins, as in PHP
            raw[lang] = d
    else:
        for lang, d in json.loads(txt).items():
            raw[lang] = {v.lstrip("#").upper(): k for k, v in d.items()}
    tables = {}
    for lang, d in raw.items():
        hexes = list(d)
        rgb = np.array([hex_to_rgb(h) for h in hexes], float)
        tables[lang] = {"names": [d[h] for h in hexes], "hex": hexes, "rgb": rgb, "lab": rgb_to_lab(rgb)}
    return tables


def nearest(rgb, table, dist):
    if dist == "rgb":  # as hexdiff() in index.php: squared RGB distance
        d = ((table["rgb"] - np.asarray(rgb, float)) ** 2).sum(1)
    else:
        d = delta_e2000(rgb_to_lab(np.array([rgb], float)), table["lab"])[0]
    j = int(np.argmin(d))
    return table["names"][j], table["hex"][j], float(d[j])


# ===== extraction

def extract_quant(im, ncolors, delta, reduce_gradients):
    w, h = im.size
    small = im.resize((150, max(1, round(h * 150 / w))), Image.BILINEAR) if w > 150 else im
    px = np.asarray(small, dtype=np.int64).reshape(-1, 3)
    if delta > 2:
        px = np.minimum(((px + delta // 2) // delta) * delta, 255)
    cols, counts = np.unique(px, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    cols, counts = cols[order], counts[order]
    if reduce_gradients:
        kept, kc = [], []
        for c, n in zip(cols, counts):
            for j, k in enumerate(kept):
                if np.max(np.abs(k - c)) <= delta:
                    kc[j] += n
                    break
            else:
                kept.append(c)
                kc.append(n)
        cols, counts = np.array(kept), np.array(kc)
        order = np.argsort(-counts, kind="stable")
        cols, counts = cols[order], counts[order]
    return cols[:ncolors].astype(float), (counts / px.shape[0])[:ncolors]


def extract_kmeans(im, k, seed=0, sample=40000, iters=50):
    small = im.copy()
    small.thumbnail((300, 300), Image.BILINEAR)
    lab = rgb_to_lab(np.asarray(small, float).reshape(-1, 3))
    rng = np.random.default_rng(seed)
    if len(lab) > sample:
        lab = lab[rng.choice(len(lab), sample, replace=False)]
    n = len(lab)
    k = min(k, n)
    C = [lab[rng.integers(n)]]
    d2 = ((lab - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        s = d2.sum()
        c = lab[rng.choice(n, p=d2 / s)] if s > 0 else lab[rng.integers(n)]
        C.append(c)
        d2 = np.minimum(d2, ((lab - c) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        lbl = ((lab[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.array([lab[lbl == j].mean(0) if np.any(lbl == j) else C[j] for j in range(k)])
        done = np.allclose(newC, C, atol=1e-3)
        C = newC
        if done:
            break
    lbl = ((lab[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    counts = np.bincount(lbl, minlength=k)
    order = [j for j in np.argsort(-counts, kind="stable") if counts[j] > 0]
    return lab_to_rgb(C[order]), counts[order] / n


# ===== pipeline

def list_inputs(p):
    p = Path(p)
    if p.is_file():
        return [p]
    return [f for f in sorted(p.iterdir())
            if f.is_file() and not f.name.startswith(".") and not f.name.startswith("liquid")
            and f.name != "index.html" and f.suffix.lower() in IMG_EXT]


def analyse(path, a, tables):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if a.mode == "quant":
        rgb, share = extract_quant(im, a.ncolors, a.delta, not a.keepgradients)
    else:
        rgb, share = extract_kmeans(im, a.k, a.seed)
    rows, index = [], {}
    for c, s in zip(rgb, share):
        hx = "".join(f"{int(v):02X}" for v in c)
        de, de_hex, de_d = nearest(c, tables["de"], a.dist)
        en, en_hex, en_d = nearest(c, tables["en"], a.dist)
        if de in index:  # index.php: if (!in_array($thiskey1,$found)) ... else skipped
            if a.dedup == "merge":
                rows[index[de]]["share"] += float(s)
            continue
        index[de] = len(rows)
        rows.append({"hex": hx, "share": float(s), "de": de, "de_hex": de_hex, "de_dist": round(de_d, 2),
                     "en": en, "en_hex": en_hex, "en_dist": round(en_d, 2)})
    return im, rows



# ===== DxO PhotoLab 10 Color-Grading recommendation

def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def lab_dir_to_hue(a, b):
    """Direction in the a*b* plane -> HSV hue (deg) of a color at L* 60, C* 40 in that direction."""
    t = np.arctan2(b, a)
    rgb = lab_to_rgb(np.array([60, 40 * np.cos(t), 40 * np.sin(t)])) / 255
    mx, mn = rgb.max(), rgb.min()
    if mx == mn:
        return 0.0
    r, g, bb = rgb
    d = mx - mn
    h = ((g - bb) / d % 6) if mx == r else ((bb - r) / d + 2) if mx == g else ((r - g) / d + 4)
    return float(h * 60 % 360)


def hue_to_lab_unit(h):
    """HSV hue (deg) -> unit vector in the a*b* plane (direction of the fully saturated color)."""
    import colorsys
    t = rgb_to_lab(np.array([[v * 255 for v in colorsys.hsv_to_rgb((h % 360) / 360, 1, 1)]]))[0]
    n = max(1e-6, float(np.hypot(t[1], t[2])))
    return np.array([t[1] / n, t[2] / n])


def tonal_weights(L):
    w_sh = _smooth((50 - L) / 30)
    w_hi = _smooth((L - 50) / 30)
    return {"allgemein": np.ones_like(L), "schatten": w_sh,
            "mitteltoene": np.clip(1 - w_sh - w_hi, 0, 1), "spitzlichter": w_hi}


def grading_recommendation(im, mode, scale=4.0, smin=15, smax=70, thresh=1.5):
    """Tonal ranges from L*: Schatten weight rises below L* 50, Spitzlichter above 50 (smoothstep over
    30 units), Mitteltöne = rest. Per range the mean a*, b* of its pixels = color cast (direction, C*).
    Modes:
      reinforce: tonal wheels toward the existing cast of that range; Allgemein 0
      neutral:   every wheel opposite to its cast
      split:     Spitzlichter toward the highlight cast (fallback: overall cast, then warm 40),
                 Schatten complementary (+180), Mitteltöne and Allgemein 0
    Sättigung = smin + scale * C*, capped at smax; casts below `thresh` C* count as neutral (0),
    except in split mode. Real photos have mean casts of only a few C* units, so a 1:1 mapping
    (previous version) gave values around 3..10 with barely visible effect in DxO. The scale is a
    heuristic and should be calibrated against DxO with the grading preview image.
    Luminanz 0 and Balance 0 are not derived from data."""
    small = im.copy()
    small.thumbnail((400, 400), Image.BILINEAR)
    lab = rgb_to_lab(np.asarray(small, float).reshape(-1, 3))
    L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
    casts = {}
    for key, w in tonal_weights(L).items():
        ws = w.sum()
        if ws < 1e-6:
            casts[key] = (0.0, 0.0, 0.0, 0.0)
            continue
        a_, b_ = float((A * w).sum() / ws), float((B * w).sum() / ws)
        casts[key] = (a_, b_, float(np.hypot(a_, b_)), round(100 * ws / len(L), 1))
    zero = {"farbton": 0, "saettigung": 0, "luminanz": 0}

    def sat(c):
        return int(np.clip(round(smin + scale * c), smin, smax))

    def entry(a_, b_, c, flip=False, extra=0.0):
        if c < thresh:
            return dict(zero)
        h = lab_dir_to_hue(-a_ if flip else a_, -b_ if flip else b_)
        return {"farbton": int(round((h + extra) % 360)), "saettigung": sat(c), "luminanz": 0}

    rec = {}
    note = None
    if mode == "reinforce":
        for k in casts:
            rec[k] = entry(*casts[k][:3])
        rec["allgemein"] = dict(zero)  # avoid doubling the tonal wheels
    elif mode == "neutral":
        for k in casts:
            rec[k] = entry(*casts[k][:3], flip=True)
    else:  # split
        src = next((casts[k] for k in ("spitzlichter", "allgemein") if casts[k][2] >= thresh), None)
        if src is None:
            h_hi, c = 40.0, 0.0
            note = "no measurable cast; split uses the conventional warm highlights / cool shadows (40 / 220)"
        else:
            h_hi, c = lab_dir_to_hue(src[0], src[1]), src[2]
        rec["spitzlichter"] = {"farbton": int(round(h_hi % 360)), "saettigung": sat(c), "luminanz": 0}
        rec["schatten"] = {"farbton": int(round((h_hi + 180) % 360)), "saettigung": sat(c), "luminanz": 0}
        rec["mitteltoene"] = dict(zero)
        rec["allgemein"] = dict(zero)
    rec["balance"] = 0
    rec["mode"] = mode
    rec["mapping"] = {"scale": scale, "min": smin, "max": smax, "threshold_C": thresh}
    rec["note"] = note
    rec["casts"] = {k: {"a": round(v[0], 2), "b": round(v[1], 2), "C": round(v[2], 2),
                        "hue_hsv": round(lab_dir_to_hue(v[0], v[1])) if v[2] > 0 else None,
                        "pixel_weight_pct": v[3]} for k, v in casts.items()}
    return rec


def grading_preview(im, rec, lab_per_point, width=1200):
    """Approximate look of the recommendation: per tonal range a Lab tint of
    Sättigung x lab_per_point chroma units in the Farbton direction, weighted by the range weight.
    DxO's own math is not public; this shows direction and relative strength, not a DxO render."""
    work = im.copy()
    if work.size[0] > width:
        work = work.resize((width, round(work.size[1] * width / work.size[0])), Image.LANCZOS)
    arr = np.asarray(work, float)
    out = np.empty_like(arr)
    for y0 in range(0, arr.shape[0], 256):
        blk = arr[y0:y0 + 256]
        lab = rgb_to_lab(blk)
        for key, w in tonal_weights(lab[..., 0]).items():
            e = rec[key]
            if e["saettigung"] <= 0:
                continue
            u = hue_to_lab_unit(e["farbton"])
            amt = e["saettigung"] * lab_per_point
            lab[..., 1] += w * amt * u[0]
            lab[..., 2] += w * amt * u[1]
        out[y0:y0 + 256] = lab_to_rgb(lab)
    return Image.fromarray(out.astype(np.uint8))


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",           # Linux
    "/System/Library/Fonts/Supplemental/Arial.ttf",              # macOS
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",                                # Windows
]


def get_font(path, size):
    from PIL import ImageFont
    for p in ([path] if path else []) + FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)  # Pillow >= 10.1, FreeType build


def label(r, lang):
    if lang == "de":
        return r["de"]
    if lang == "en":
        return r["en"]
    return f'{r["de"]} => {r["en"]}'


def legend_image(im, rows, a, title, grading=None):
    """Image on top; below a proportional color strip (share of all pixels, rest in light gray)
    and a legend grid: swatch, name, hex, share."""
    from PIL import ImageDraw
    W = a.width
    top = im.copy()
    top = top.resize((W, max(1, round(im.size[1] * W / im.size[0]))), Image.LANCZOS)
    fs = max(12, round(W / 60))
    font = get_font(a.font, fs)
    small = get_font(a.font, max(10, round(fs * 0.8)))
    pad = round(fs * 0.9)
    strip_h = round(fs * 2.2)
    cols = a.cols if a.cols else max(1, min(4, W // (fs * 22)))
    rest = max(0.0, 1.0 - sum(r["share"] for r in rows))
    entries = [(r["hex"], label(r, a.lang), f'#{r["hex"]}   {r["share"] * 100:.1f} %') for r in rows]
    if rest >= 0.005:
        entries.append(("DDDDDD", "Rest" if a.lang != "en" else "rest", f"{rest * 100:.1f} % (not listed)"))
    row_h = round(fs * 2.9)
    nrows = -(-len(entries) // cols)
    title_h = round(fs * 1.8) if title else 0
    grad_h = round(fs * (2.2 + 5 * 1.5 + 3.4)) if grading else 0
    H = top.size[1] + pad + strip_h + pad + nrows * row_h + pad + grad_h + title_h
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    canvas.paste(top, (0, 0))
    d = ImageDraw.Draw(canvas)
    y = top.size[1] + pad
    x = 0.0
    for r in rows:  # strip, widths = share of all pixels
        w = r["share"] * W
        d.rectangle([round(x), y, round(x + w), y + strip_h], fill="#" + r["hex"])
        x += w
    if x < W:
        d.rectangle([round(x), y, W, y + strip_h], fill="#DDDDDD")
    d.rectangle([0, y, W - 1, y + strip_h], outline=(150, 150, 150))  # makes white segments visible
    y += strip_h + pad
    colw = W / cols
    sw = round(fs * 1.9)
    for k, (hx, name, sub) in enumerate(entries):
        cx = round((k % cols) * colw) + pad
        cy = y + (k // cols) * row_h
        d.rectangle([cx, cy, cx + sw, cy + sw], fill="#" + hx, outline=(150, 150, 150))
        d.text((cx + sw + round(fs * 0.5), cy - round(fs * 0.1)), name, font=font, fill=(20, 20, 20))
        d.text((cx + sw + round(fs * 0.5), cy + round(fs * 1.15)), sub, font=small, fill=(110, 110, 110))
    if grading:
        gy = y + nrows * row_h + pad
        names = {"reinforce": "Stimmung verstärken", "neutral": "Farbstich neutralisieren",
                 "split": "Split (komplementär)"}
        d.line([pad, gy, W - pad, gy], fill=(200, 200, 200))
        d.text((pad, gy + round(fs * 0.4)), f"DxO PhotoLab 10 Color-Grading, Empfehlung: {names[grading['mode']]}",
               font=font, fill=(20, 20, 20))
        ty = gy + round(fs * 2.2)
        for j, (key, lab_) in enumerate((("allgemein", "Allgemein"), ("schatten", "Schatten"),
                                         ("mitteltoene", "Mitteltöne"), ("spitzlichter", "Spitzlichter"))):
            e = grading[key]
            d.text((pad, ty + round(fs * 1.5 * j)), lab_, font=font, fill=(20, 20, 20))
            d.text((pad + round(fs * 9), ty + round(fs * 1.5 * j)),
                   f"Farbton {e['farbton']}    Sättigung {e['saettigung']}    Luminanz {e['luminanz']}",
                   font=font, fill=(60, 60, 60))
        d.text((pad, ty + round(fs * 1.5 * 4)), "Balance", font=font, fill=(20, 20, 20))
        d.text((pad + round(fs * 9), ty + round(fs * 1.5 * 4)), "0 (Mitte)", font=font, fill=(60, 60, 60))
        ny = ty + round(fs * 1.5 * 5) + round(fs * 0.3)  # note below the Balance line
        m = grading["mapping"]
        note = (f"Farbton = Winkel ab Rot gegen den Uhrzeigersinn. Sättigung = {m['min']} + {m['scale']:g} x "
                f"Farbstich C* des Tonbereichs, max. {m['max']} (Heuristik, DxO-Skalierung nicht kalibriert). "
                + ("Kein messbarer Farbstich: Split nach Konvention warm/kühl. " if grading.get("note") else ""))
        lines, cur = [], ""
        for word in note.split():
            if d.textlength((cur + " " + word).strip(), font=small) > W - 2 * pad:
                lines.append(cur)
                cur = word
            else:
                cur = (cur + " " + word).strip()
        lines.append(cur)
        for j, ln in enumerate(lines):
            d.text((pad, ny + round(fs * 1.1 * j)), ln, font=small, fill=(120, 120, 120))
    if title:
        d.text((pad, H - title_h + round(fs * 0.3)), title, font=small, fill=(140, 140, 140))
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Dominant colors with German (default) or English names")
    ap.add_argument("-in", dest="inp", required=True, help="image file or directory")
    ap.add_argument("-names", required=True, help="colors.names.php or JSON")
    ap.add_argument("-out", default="colorcode_out")
    ap.add_argument("-mode", choices=["quant", "kmeans"], default="quant")
    ap.add_argument("-ncolors", type=int, default=15, help="quant: num_results")
    ap.add_argument("-delta", type=int, default=16, help="quant: rounding step")
    ap.add_argument("-keepgradients", action="store_true", help="quant: reduce_gradients=false")
    ap.add_argument("-k", type=int, default=8, help="kmeans: clusters")
    ap.add_argument("-seed", type=int, default=0, help="kmeans: RNG seed")
    ap.add_argument("-dist", choices=["rgb", "de2000"], default="rgb", help="name matching distance")
    ap.add_argument("-dedup", choices=["skip", "merge"], default="skip",
                    help="repeated German name: skip (PHP) or add its share to the first")
    ap.add_argument("-lang", choices=["de", "en", "both"], default="de", help="legend language")
    ap.add_argument("-width", type=int, default=1600, help="output width in px")
    ap.add_argument("-cols", type=int, default=0, help="legend columns, 0 = automatic")
    ap.add_argument("-font", help="TrueType font file (default: DejaVu Sans / Arial if found)")
    ap.add_argument("-format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("-grading", choices=["reinforce", "neutral", "split", "off"], default="reinforce",
                    help="DxO Color-Grading recommendation below the legend")
    ap.add_argument("-cgscale", type=float, default=4.0, help="Sättigung = cgmin + cgscale x cast C*")
    ap.add_argument("-cgmin", type=int, default=15, help="minimum Sättigung for a non-neutral range")
    ap.add_argument("-cgmax", type=int, default=70, help="maximum Sättigung")
    ap.add_argument("-cglab", type=float, default=0.25,
                    help="preview only: Lab chroma units per Sättigung point (uncalibrated)")
    ap.add_argument("-nopreview", action="store_true", help="no <stem>_grading_preview.jpg")
    ap.add_argument("-notitle", action="store_true", help="omit file name and settings line")
    a = ap.parse_args()

    tables = load_names(a.names)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = list_inputs(a.inp)
    if not files:
        sys.exit("no images found")

    allrows = []
    settings = (f"mode {a.mode}, " + (f"{a.ncolors} colors, delta {a.delta}, gradients "
                                      f"{'kept' if a.keepgradients else 'merged'}" if a.mode == "quant"
                                      else f"k {a.k}, seed {a.seed}") + f", names {a.dist}, duplicates {a.dedup}")
    for f in files:
        try:
            im, rows = analyse(f, a, tables)
        except Exception as ex:
            print(f"fail {f.name}: {ex}", file=sys.stderr)
            continue
        grading = None if a.grading == "off" else grading_recommendation(im, a.grading, a.cgscale, a.cgmin, a.cgmax)
        if grading and not a.nopreview:
            grading_preview(im, grading, a.cglab).save(out / f"{f.stem}_grading_preview.jpg", quality=90)
        img = legend_image(im, rows, a, None if a.notitle else f"{f.name}   ({settings})", grading)
        target = out / f"{f.stem}_colors.{a.format}"
        if a.format == "jpg":
            img.save(target, quality=92)
        else:
            img.save(target)
        (out / f"{f.stem}.json").write_text(json.dumps({"file": f.name, "settings": settings, "colors": rows,
                                                        "dxo_color_grading": grading},
                                                       indent=2, ensure_ascii=False), encoding="utf-8")
        for i, r in enumerate(rows, 1):
            allrows.append([f.name, i, r["hex"], round(r["share"], 4), r["de"], r["de_dist"], r["en"], r["en_dist"]])
        print(f"ok   {target.resolve()}: " + ", ".join(f"{label(r, a.lang)} {r['share'] * 100:.0f}%" for r in rows[:5])
              + (" ..." if len(rows) > 5 else ""))
        if grading:
            print("     DxO Color-Grading (" + grading["mode"] + "): " + "; ".join(
                f"{n} Farbton {grading[k]['farbton']} Sättigung {grading[k]['saettigung']}"
                for k, n in (("allgemein", "Allgemein"), ("schatten", "Schatten"), ("mitteltoene", "Mitteltöne"),
                             ("spitzlichter", "Spitzlichter"))) + "; Balance 0")

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "rank", "hex", "share", "de", f"de_{a.dist}", "en", f"en_{a.dist}"])
        w.writerows(allrows)
    print(f"summary: {(out / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
