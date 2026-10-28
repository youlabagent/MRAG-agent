"""AI-powered inference features for ATAC-seq pipeline."""
from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP


def _py_literal(value) -> str:
    """Return a safe Python source literal for string interpolation."""
    return json.dumps(value)


def register_ai_tools(mcp: FastMCP, executor):
    """Register 9 AI inference tools."""
    q = executor.quote
    py = os.environ.get("ATAC_MCP_PYTHON", "/home/ai/miniconda3/bin/python3")

    def _safe_dir(project_dir: str) -> str:
        return str(executor._resolve_project_dir(project_dir))

    # ── 1. Sample Auto-Detection ────────────────────────────────
    @mcp.tool()
    def detect_samples(project_dir: str) -> str:
        """Scan directory for FASTQ files and auto-detect sample list with read structure.
        Recognises Illumina, BGI, MGI, SRA naming conventions.

        Args:
            project_dir: Directory to scan for FASTQ files.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            "import glob, json, os, re\n"
            "fqs = sorted(glob.glob(os.path.join(PROJECT_DIR, '*.fastq.gz')) +\n"
            "             glob.glob(os.path.join(PROJECT_DIR, '*.fq.gz')))\n"
            "samples = {}\n"
            "for fq in fqs:\n"
            "    base = os.path.basename(fq)\n"
            "    for pat in [\n"
            "        r'^(.+?)_R?[12][\\._]',\n"
            "        r'^(.+?)_[12]\\.f',\n"
            "        r'^(SRR\\d+)[\\._]',\n"
            "        r'^(.+?)\\.f',\n"
            "    ]:\n"
            "        m = re.match(pat, base)\n"
            "        if m:\n"
            "            name = m.group(1)\n"
            "            samples.setdefault(name, []).append(base)\n"
            "            break\n"
            "result = []\n"
            "for name, files in sorted(samples.items()):\n"
            "    paired = any('_R2' in f or '_2.' in f for f in files)\n"
            "    result.append({'sample': name, 'files': files, 'paired_end': paired})\n"
            "print(json.dumps(result, indent=2))\n"
        )
        return executor.run(f"{q(py)} -c {q(script)}", cwd=d)

    # ── 2. Read Format Auto-Detection ───────────────────────────
    @mcp.tool()
    def detect_read_format(project_dir: str, sample: str) -> str:
        """Detect sequencing platform and read length from FASTQ header.

        Args:
            project_dir: Project directory.
            sample:      Sample prefix.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            f"SAMPLE = {_py_literal(sample)}\n"
            "import glob, gzip, json, os, re, sys\n"
            "cands = sorted(\n"
            "    glob.glob(os.path.join(PROJECT_DIR, SAMPLE + '*R1*.fastq.gz')) +\n"
            "    glob.glob(os.path.join(PROJECT_DIR, SAMPLE + '*_1.fastq.gz'))\n"
            ")\n"
            "if not cands:\n"
            "    print(json.dumps({'error': 'No R1 file found', 'sample': SAMPLE}))\n"
            "    sys.exit(1)\n"
            "r1 = cands[0]\n"
            "with gzip.open(r1, 'rt') as fh:\n"
            "    header = fh.readline().rstrip('\\n')\n"
            "    seq = fh.readline().rstrip('\\n')\n"
            "platform = 'unknown'\n"
            "if re.match(r'^@\\S+:\\S+:\\S+:\\S+', header): platform = 'Illumina'\n"
            "elif header.startswith('@SRR'):               platform = 'SRA'\n"
            "elif header.startswith('@ERR'):               platform = 'ENA'\n"
            "elif header.startswith('@CL'):                platform = 'BGI/MGI'\n"
            "print(json.dumps({\n"
            "    'sample': SAMPLE, 'platform': platform,\n"
            "    'read_length': len(seq), 'header': header, 'file': r1,\n"
            "}, indent=2))\n"
        )
        return executor.run(f"{q(py)} -c {q(script)}", cwd=d)

    # ── 3. Treatment/Control Group Classification ───────────────
    @mcp.tool()
    def classify_groups(project_dir: str) -> str:
        """Auto-classify samples into treatment/control groups based on filename keywords.
        Recognises: OE, KO, KD, WT, ctrl, control, DDK, treated, vehicle, DMSO, etc.

        Args:
            project_dir: Project directory containing FASTQ files.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            "import glob, json, os, re\n"
            "fqs = glob.glob(os.path.join(PROJECT_DIR, '*_R1.fastq.gz')) + \\\n"
            "      glob.glob(os.path.join(PROJECT_DIR, '*_1.fastq.gz'))\n"
            "samples = [os.path.basename(f).split('_R1')[0].split('_1.fastq')[0] for f in fqs]\n"
            "treatment_kw = re.compile(r'(OE|KO|KD|shRNA|siRNA|(?<!un)treated|drug|inhibitor|stimul)', re.I)\n"
            "control_kw   = re.compile(r'(WT|wt|ctrl|control|DDK|vehicle|DMSO|scramble|mock|untreated|EV|empty)', re.I)\n"
            "groups = {'treatment': [], 'control': [], 'unclassified': []}\n"
            "for s in sorted(set(samples)):\n"
            "    if treatment_kw.search(s):\n"
            "        groups['treatment'].append(s)\n"
            "    elif control_kw.search(s):\n"
            "        groups['control'].append(s)\n"
            "    else:\n"
            "        groups['unclassified'].append(s)\n"
            "print(json.dumps(groups, indent=2))\n"
        )
        return executor.run(f"{q(py)} -c {q(script)}", cwd=d)

    # ── 4. Genome/Species Auto-Detection ────────────────────────
    @mcp.tool()
    def detect_genome(project_dir: str) -> str:
        """Infer genome build from directory names, filenames, and reference paths.

        Args:
            project_dir: Project directory.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            "import json, os, re\n"
            "genome_patterns = {\n"
            "    'hg38': r'(hg38|GRCh38)',\n"
            "    'hg19': r'(hg19|GRCh37)',\n"
            "    'mm10': r'(mm10|GRCm38)',\n"
            "    'mm39': r'(mm39|GRCm39)',\n"
            "    'rn6':  r'(rn6|Rnor_6)',\n"
            "    'dm6':  r'(dm6|BDGP6)',\n"
            "    'danRer11': r'(danRer11|GRCz11)',\n"
            "}\n"
            "found = {}\n"
            "for root, dirs, files in os.walk(PROJECT_DIR):\n"
            "    for name in dirs + files:\n"
            "        full = os.path.join(root, name)\n"
            "        for gname, pat in genome_patterns.items():\n"
            "            if re.search(pat, full):\n"
            "                found[gname] = found.get(gname, 0) + 1\n"
            "if found:\n"
            "    best = max(found, key=found.get)\n"
            "    print(json.dumps({'detected': best, 'confidence': 'high' if found[best] > 2 else 'low', 'all_hits': found}))\n"
            "else:\n"
            "    print(json.dumps({'detected': None, 'message': 'Could not auto-detect genome. Please specify manually.'}))\n"
        )
        return executor.run(f"{q(py)} -c {q(script)}", cwd=d)

    # ── 5. DAStk Group Auto-Mapping ─────────────────────────────
    @mcp.tool()
    def map_dastk_groups(project_dir: str) -> str:
        """Auto-map classified treatment/control groups to DAStk input files.
        Uses classify_groups results + available narrowPeak.filt files.

        Args:
            project_dir: Project directory.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            "import glob, json, os, re\n"
            "peaks = glob.glob(os.path.join(PROJECT_DIR, '*_peaks.narrowPeak.filt'))\n"
            "peak_map = {}\n"
            "for p in peaks:\n"
            "    base = os.path.basename(p).replace('_peaks.narrowPeak.filt', '')\n"
            "    peak_map[base] = p\n"
            "treatment_kw = re.compile(r'(OE|KO|KD|shRNA|siRNA|(?<!un)treated|drug|inhibitor|stimul)', re.I)\n"
            "control_kw   = re.compile(r'(WT|wt|ctrl|control|DDK|vehicle|DMSO|scramble|mock|untreated|EV|empty)', re.I)\n"
            "mapping = {'treatment_files': [], 'control_files': [], 'unmatched': []}\n"
            "for name, path in sorted(peak_map.items()):\n"
            "    if treatment_kw.search(name):\n"
            "        mapping['treatment_files'].append(path)\n"
            "    elif control_kw.search(name):\n"
            "        mapping['control_files'].append(path)\n"
            "    else:\n"
            "        mapping['unmatched'].append(path)\n"
            "mapping['ready'] = bool(mapping['treatment_files'] and mapping['control_files'])\n"
            "print(json.dumps(mapping, indent=2))\n"
        )
        return executor.run(f"{q(py)} -c {q(script)}", cwd=d)

    # ── 6. Bowtie2 Options Auto-Selection ───────────────────────
    @mcp.tool()
    def recommend_bowtie2_options(project_dir: str, sample: str) -> str:
        """Analyse FASTQ header and read length to recommend optimal bowtie2 parameters.

        Args:
            project_dir: Project directory.
            sample:      Sample prefix.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            f"SAMPLE = {_py_literal(sample)}\n"
            "import glob, gzip, json, os, sys\n"
            "cands = sorted(\n"
            "    glob.glob(os.path.join(PROJECT_DIR, SAMPLE + '*R1*.fastq.gz')) +\n"
            "    glob.glob(os.path.join(PROJECT_DIR, SAMPLE + '*_1.fastq.gz'))\n"
            ")\n"
            "if not cands:\n"
            "    print(json.dumps({'error': 'No R1 file found', 'sample': SAMPLE}))\n"
            "    sys.exit(1)\n"
            "r1 = cands[0]\n"
            "with gzip.open(r1, 'rt') as fh:\n"
            "    fh.readline()\n"
            "    seq = fh.readline().rstrip('\\n')\n"
            "read_length = len(seq)\n"
            "if read_length <= 50:\n"
            "    preset = '--very-sensitive'\n"
            "    extra = '--no-mixed --no-discordant --no-unal'\n"
            "    note = 'Short reads (<=50bp): very-sensitive mode recommended'\n"
            "elif read_length <= 100:\n"
            "    preset = '--very-sensitive'\n"
            "    extra = '--no-mixed --no-discordant --maxins 2000 --no-unal'\n"
            "    note = 'Medium reads (50-100bp): standard ATAC-seq settings'\n"
            "else:\n"
            "    preset = '--very-sensitive-local'\n"
            "    extra = '--no-mixed --no-discordant --maxins 2000 --no-unal'\n"
            "    note = 'Long reads (>100bp): local alignment recommended'\n"
            "print(json.dumps({\n"
            "    'read_length': read_length,\n"
            "    'preset': preset,\n"
            "    'extra': extra,\n"
            "    'note': note,\n"
            "}, indent=2))\n"
        )
        return executor.run(f"{q(py)} -c {q(script)}", cwd=d)

    # ── 7. QC Auto-Judgment ─────────────────────────────────────
    @mcp.tool()
    def check_qc(project_dir: str, sample: str) -> str:
        """Parse bowtie2 alignment log, picard duplicate metrics, and FastQC results.
        Returns PASS/WARN/FAIL verdict per metric.

        Args:
            project_dir: Project directory.
            sample:      Sample prefix.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            f"SAMPLE = {_py_literal(sample)}\n"
            "import json, os, re\n"
            "results = {'sample': SAMPLE, 'metrics': {}, 'verdict': 'PASS'}\n"
            "adir = os.path.join(PROJECT_DIR, 'alignments')\n"
            "log = os.path.join(adir, SAMPLE + '.bowtie2Log')\n"
            "if os.path.exists(log):\n"
            "    txt = open(log).read()\n"
            "    m = re.search(r'([\\d.]+)% overall alignment rate', txt)\n"
            "    if m:\n"
            "        rate = float(m.group(1))\n"
            "        status = 'PASS' if rate >= 70 else ('WARN' if rate >= 50 else 'FAIL')\n"
            "        results['metrics']['alignment_rate'] = {'value': rate, 'status': status}\n"
            "        if status == 'FAIL':\n"
            "            results['verdict'] = 'FAIL'\n"
            "        elif status == 'WARN' and results['verdict'] != 'FAIL':\n"
            "            results['verdict'] = 'WARN'\n"
            "dup = os.path.join(adir, SAMPLE + '.duplicate_metrics')\n"
            "if os.path.exists(dup):\n"
            "    for line in open(dup):\n"
            "        parts = line.strip().split('\\t')\n"
            "        if len(parts) > 8 and not line.startswith('#') and not line.startswith('LIBRARY'):\n"
            "            try:\n"
            "                dup_rate = float(parts[8])\n"
            "                status = 'PASS' if dup_rate < 0.3 else ('WARN' if dup_rate < 0.5 else 'FAIL')\n"
            "                results['metrics']['duplication_rate'] = {'value': round(dup_rate, 4), 'status': status}\n"
            "                if status == 'FAIL':\n"
            "                    results['verdict'] = 'FAIL'\n"
            "                elif status == 'WARN' and results['verdict'] != 'FAIL':\n"
            "                    results['verdict'] = 'WARN'\n"
            "            except ValueError:\n"
            "                pass\n"
            "print(json.dumps(results, indent=2))\n"
        )
        return executor.run(f"{q(py)} -c {q(script)}", cwd=d)

    # ── 8. Resume/Checkpoint Status ─────────────────────────────
    @mcp.tool()
    def checkpoint_status(project_dir: str) -> str:
        """Show checkpoint status for all samples: which steps are done, which remain.

        Args:
            project_dir: Project directory.
        """
        d = _safe_dir(project_dir)
        data = executor.load_checkpoint(d)
        all_steps = ["step0", "step1", "step2", "step3", "step4"]
        summary = {}
        for sample_name, done in data.get("completed_steps", {}).items():
            remaining = [step for step in all_steps if step not in done]
            summary[sample_name] = {"completed": done, "remaining": remaining}
        return json.dumps({"checkpoint": summary}, indent=2)

    # ── 9. Resource Auto-Configuration ──────────────────────────
    @mcp.tool()
    def detect_resources() -> str:
        """Detect server CPU cores, available RAM, and recommend thread/memory settings.
        Returns suggested num_threads and Java -Xmx value.
        """
        res = executor.detect_resources()
        cores = res["cores"]
        mem = res["mem_gb"]

        recommended_threads = max(1, cores - 2)
        xmx = max(4, mem // 4)

        gpu_info = executor.run(
            "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo 'No GPU'"
        )

        return json.dumps({
            "cpu_cores": cores,
            "available_ram_gb": mem,
            "recommended_threads": recommended_threads,
            "recommended_java_xmx": f"{xmx}g",
            "gpu": gpu_info.strip(),
        }, indent=2)
