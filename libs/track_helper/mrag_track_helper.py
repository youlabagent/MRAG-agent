"""
MRAG ATAC track 시각화 helper.

핵심 기능:
- 여러 sample을 한 region에 stacked로 그려서 PNG/PDF 생성
- y축 max_value를 region 데이터에 맞춰 자동 계산 (모든 sample 공유)
- rep끼리 같은 색 family로 자동 매핑

사용 예:
    from mrag_track_helper import plot_atac_tracks

    out = plot_atac_tracks(
        samples=[("ctrl_1", "/ctrl1.bw", "/ctrl1.np"),
                 ("ctrl_2", "/ctrl2.bw", "/ctrl2.np"),
                 ("OE_1",   "/OE1.bw",   "/OE1.np"),
                 ("OE_2",   "/OE2.bw",   "/OE2.np")],
        region="chr8:127720000-127750000",
        title="MYC_locus",
        max_value="shared_auto",   # 또는 "q99" / "q95" / 숫자 / "auto"
    )
"""
from __future__ import annotations
import subprocess
import re
from pathlib import Path
from typing import Sequence, Tuple, Union

# The HUMAN gene models. It is a DEFAULT, not a constant: most mm10
# coordinates fall inside hg38 chromosome bounds, so on a mouse run tabix
# succeeds and pyGenomeTracks quietly draws whichever unrelated human genes sit
# at those coordinates, under a caption naming the correct mouse gene (measured
# 2026-09-23: 87 of 150 mouse loci have a human gene at the same coordinates).
# Callers that know the organism MUST pass gene_gtf.
GENE_GTF_DEFAULT = "/BIO1/MRAG/Agent/Reference/RNA_hg38/gencode.v34.pc_lncRNA.sorted.gtf.gz"

import numpy as np
import pyBigWig


# 색상 팔레트 (condition별로 한 색 family, 형제 rep는 명도만 다름)
# House convention (2026-07): treatment = RED, control = BLUE, always.
# Callers pass `colors` ordered to match the sample list so the mapping follows
# the arm ROLE, not its position; this list is only the positional fallback.
_DEFAULT_COLORS = [
    ("#d62728", "#e57878"),   # red   (treatment)
    ("#1f77b4", "#4a90d9"),   # blue  (control)
    ("#2ca02c", "#5fbf5f"),   # green
    ("#9467bd", "#b894d2"),   # purple
    ("#ff7f0e", "#ffa856"),   # orange
]

MaxValueSpec = Union[float, int, str, None]


def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.strip()).strip("_").lower() or "region"


def _parse_region(region: str) -> tuple[str, int, int]:
    """ 'chr8:127,720,000-127,750,000' → ('chr8', 127720000, 127750000) """
    chrom, span = region.split(":")
    start, end = span.replace(",", "").split("-")
    return chrom.strip(), int(start), int(end)


def _compute_shared_max(
    bigwig_paths: Sequence[str],
    region: str,
    method: str = "shared_auto",
) -> float:
    """
    region 안에서 여러 bigwig의 신호를 읽어 공통 max_value 계산.

    method:
        "shared_auto" - 모든 sample 평균 max에 약간 margin
        "shared_max"  - 전체 max (outlier에 민감)
        "q99"/"q95"/"q90" - 해당 quantile
    """
    chrom, start, end = _parse_region(region)
    all_vals = []
    per_sample_max = []

    for bw_path in bigwig_paths:
        bw = pyBigWig.open(bw_path)
        try:
            if chrom not in bw.chroms():
                continue
            v = bw.values(chrom, start, end, numpy=True)
            v = v[~np.isnan(v)]
            if v.size == 0:
                continue
            all_vals.append(v)
            per_sample_max.append(float(v.max()))
        finally:
            bw.close()

    if not all_vals:
        return 30.0  # fallback

    concatenated = np.concatenate(all_vals)

    if method == "shared_max":
        max_v = float(concatenated.max())
    elif method.startswith("q"):
        try:
            q = float(method[1:]) / 100
        except ValueError:
            q = 0.99
        max_v = float(np.quantile(concatenated, q))
    else:  # "shared_auto" 기본 — sample별 max의 median + 10% headroom
        max_v = float(np.median(per_sample_max)) * 1.1

    # 시각화에 의미 있는 최소값 보장 (너무 작으면 의미 없음)
    max_v = max(max_v, 1.0)
    # 보기 좋게 round up to nearest 5
    max_v = float(int(np.ceil(max_v / 5.0)) * 5)
    return max_v


def _build_ini(
    samples: Sequence[Tuple[str, str, str | None]],
    *,
    max_value: float | str | None,
    colors: Sequence[Tuple[str, str]] | None = None,
    gene_gtf: str | None = GENE_GTF_DEFAULT,
    gene_title: str = "genes",
    highlight_bed: str | None = None,
) -> str:
    """tracks.ini 문자열 생성."""
    colors = colors or _DEFAULT_COLORS
    lines = []
    # Shaded vertical band over the differential peak site (spans all tracks).
    if highlight_bed:
        lines += [
            "[peak_highlight]",
            f"file = {highlight_bed}",
            "type = vhighlight",
            "color = #f6c343",
            "alpha = 0.12",
            "",
        ]
    # Gene-model track at the very top: exon boxes + intron arrows
    # (strand/promoter direction), tabix-queried for speed.
    if gene_gtf:
        lines += [
            "[genes]",
            f"file = {gene_gtf}",
            f"title = {gene_title}",
            "file_type = gtf",
            "height = 1.3",
            "gene_rows = 2",
            "fontsize = 9",
            "color = #08306b",
            "border_color = #08306b",
            "prefered_name = gene_name",
            "merge_transcripts = true",
            "merge_overlapping_exons = true",
            "style = UCSC",
            "arrow_interval = 5",
            "arrowhead_included = true",
            "labels = true",
            "all_labels_inside = true",
            "",
        ]
    lines += [
        "[spacer]",
        "height = 0.3",
        "",
    ]

    n = len(samples)
    for i, (name, bw, peak) in enumerate(samples):
        condition_idx = i // 2 if n >= 2 else 0
        rep_idx = i % 2
        family = colors[condition_idx % len(colors)]
        color = family[0]  # uniform color per condition (both reps same hue)

        block = [
            f"[{name} bigwig]",
            f"file = {bw}",
            f"title = {name}",
            f"color = {color}",
            "height = 0.75",
            "min_value = 0",
            "show_data_range = true",
            "nans_to_zeros = true",
        ]
        if max_value is not None:
            if isinstance(max_value, str):
                block.append(f"max_value = {max_value}")  # 'auto' 같은 keyword
            else:
                block.append(f"max_value = {max_value}")
        block.append("")
        lines += block

        if peak:
            lines += [
                f"[{name} peaks]",
                f"file = {peak}",
                f"title = peaks",
                "file_type = narrow_peak",
                f"color = {color}",
                "height = 0.8",
                "show_labels = false",
                "show_data_range = false",
                "use_summit = false",
                "",
            ]
        lines += ["[spacer]", "height = 0.15", ""]

    lines += ["[x-axis]", "where = bottom", "fontsize = 10"]
    return "\n".join(l for l in lines if l is not None)


def plot_atac_tracks(
    samples: Sequence[Tuple[str, str, str | None]],
    region: str,
    *,
    title: str | None = None,
    out_dir: str = "/BIO1/MRAG/Agent/test_viz/tracks",
    formats: Sequence[str] = ("png", "pdf"),
    max_value: MaxValueSpec = "shared_max",
    dpi: int = 200,
    width: float = 24,
    height: float | None = None,
    binary: str = "/home/ai/.local/bin/pyGenomeTracks",
    gene_title: str = "genes",
    font_size: int = 6,
    highlight: tuple | None = None,
    colors: Sequence[Tuple[str, str]] | None = None,
    gene_gtf: str | None = GENE_GTF_DEFAULT,
) -> dict:
    """
    여러 ATAC sample을 한 region에 stacked로 그려서 PNG/PDF로 저장.

    Args:
        samples: [(label, bigwig_path, narrowpeak_path_or_None), ...]
        region:  "chr8:127720000-127750000" 형식
        title:   파일명에 들어가는 짧은 라벨
        out_dir: 출력 디렉토리
        formats: 생성할 포맷 (png/pdf/svg)
        max_value: y축 max 설정 — 핵심 옵션
            - "shared_auto" (기본): region 데이터에서 sample별 max의 median × 1.1.
                                    모든 sample이 같은 max 공유 → 비교 가능
            - "shared_max":  모든 sample의 region max (outlier 민감)
            - "q99"/"q95"/"q90"/"q75": 해당 percentile
            - "auto":        pyGenomeTracks에 위임 (track별 독립 — 비교 불가)
            - 숫자 (예: 30): 고정값
            - None:          pyGenomeTracks 기본
        dpi, width, height: matplotlib 인자
        binary: pyGenomeTracks 실행 파일 경로

    Returns:
        {"ini", "png", "pdf", "max_value_used", "max_value_method"} dict
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    label = _slugify(title or region.replace(":", "_"))
    ini_path = out / f"mrag_tracks_{label}.ini"

    if height is None:
        # Tight fit: gene track (1.3) + bigwig (0.75 each) + spacer (0.4)
        #            + top spacer (0.3) + xaxis (~1)
        height = 1.3 + 0.3 + len(samples) * (0.75 + 0.4) + 1.0

    # max_value 자동 계산
    method_used = "fixed"
    computed_max = max_value
    if isinstance(max_value, str) and max_value.startswith(("shared_", "q")):
        bw_paths = [s[1] for s in samples]
        computed_max = _compute_shared_max(bw_paths, region, method=max_value)
        method_used = max_value

    # Shaded peak highlight band permanently disabled per user request.
    highlight_bed = None
    # ini 작성
    ini_text = _build_ini(samples, max_value=computed_max, gene_title=gene_title,
                          highlight_bed=highlight_bed, colors=colors,
                          gene_gtf=gene_gtf)
    ini_path.write_text(ini_text)

    result = {
        "ini": str(ini_path),
        "max_value_used": computed_max,
        "max_value_method": method_used,
        "region": region,
    }

    for fmt in formats:
        out_path = out / f"mrag_{label}.{fmt}"
        cmd = [
            binary,
            "--tracks", str(ini_path),
            "--region", region,
            "--outFileName", str(out_path),
            "--dpi", str(dpi),
            "--width", str(width),
            "--height", str(height),
            "--fontSize", str(font_size),
            "--trackLabelFraction", "0.20",
            "--trackLabelHAlign", "left",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"pyGenomeTracks failed for {fmt}:\n{proc.stderr[-800:]}"
            )
        result[fmt] = str(out_path)

    return result
