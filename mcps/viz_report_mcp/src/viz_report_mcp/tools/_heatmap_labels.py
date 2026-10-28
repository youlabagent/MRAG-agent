"""PIL post-processor: add per-row gene labels to deepTools heatmap PNG.

deepTools plotHeatmap doesn't natively support per-row labels. We:
  1. Open the PNG
  2. Detect heatmap row area by scanning for colored pixels
  3. Read sorted_regions.bed (deepTools post-clustering order)
  4. Draw gene names along the right edge
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


def _read_sorted_genes(bed_path: str) -> List[str]:
    out = []
    with open(bed_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) >= 4: out.append(parts[3])
    return out


def _font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try: return ImageFont.truetype(c, size=size)
            except Exception: continue
    return ImageFont.load_default()


def _detect_heatmap_area(img: Image.Image) -> Tuple[int, int, int, int]:
    """Return (y_top, y_bottom, x_right_panel, label_x_start) — rows of heatmap.

    Method: scan a vertical column near the right side. Heatmap rows have
    non-white pixels (colored gradient). Find the contiguous block.
    """
    px = img.load()
    W, H = img.size
    # x at roughly 0.8 of width is near the rightmost panel (last sample)
    x_scan = int(W * 0.75)
    band = []
    for y in range(H):
        r, g, b = px[x_scan, y][:3]
        non_white = (r < 245 or g < 245 or b < 245) and not (r > 200 and g < 200 and b < 200 and r-g > 50)  # exclude red title text
        if non_white:
            band.append(y)
    if not band:
        # fallback: assume middle 70%
        return int(H * 0.18), int(H * 0.85), W - 50, W - 45
    # Pick largest contiguous block
    runs = []
    cur = [band[0]]
    for y in band[1:]:
        if y == cur[-1] + 1: cur.append(y)
        else: runs.append(cur); cur = [y]
    runs.append(cur)
    biggest = max(runs, key=len)
    y_top, y_bot = biggest[0], biggest[-1]

    # Find right edge of rightmost panel — scan from right inward at the row middle
    y_mid = (y_top + y_bot) // 2
    x_right = W - 1
    for x in range(W - 1, int(W * 0.5), -1):
        r, g, b = px[x, y_mid][:3]
        if (r < 245 or g < 245 or b < 245):
            x_right = x
            break
    return y_top, y_bot, x_right, min(x_right + 8, W - 5)


def add_gene_labels(
    heatmap_png: str,
    sorted_regions_bed: str,
    output_png: str,
    *,
    font_size: int = 9,
    color: str = "#222",
    max_genes: int = 0,  # 0 = all
):
    genes = _read_sorted_genes(sorted_regions_bed)
    if max_genes and len(genes) > max_genes:
        genes = genes[:max_genes]
    img = Image.open(heatmap_png).convert("RGBA")
    y_top, y_bot, x_panel, x_lbl = _detect_heatmap_area(img)
    n = len(genes)
    if n == 0:
        Path(output_png).write_bytes(Path(heatmap_png).read_bytes())
        return
    row_h = (y_bot - y_top) / max(n, 1)
    # If labels would be unreadable (font > row spacing), shrink font
    fs = min(font_size, max(7, int(row_h * 1.0)))
    font = _font(fs)

    # Extend canvas if labels would overflow right edge
    needed = max(font.getbbox(g)[2] for g in genes) + 16
    W, H = img.size
    if x_lbl + needed > W:
        new = Image.new("RGBA", (x_lbl + needed + 4, H), (255, 255, 255, 255))
        new.paste(img, (0, 0))
        img = new
        W, H = img.size

    draw = ImageDraw.Draw(img)
    for i, g in enumerate(genes):
        y = y_top + (i + 0.5) * row_h - fs * 0.55
        if y < 0 or y > H - fs: continue
        draw.text((x_lbl, y), g, fill=color, font=font)

    img.convert("RGB").save(output_png)
