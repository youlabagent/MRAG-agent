"""AI-powered inference features for RNA-seq pipeline (11 tools).

Security notes:
- User inputs (project_dir, sample) are injected into `python -c` scripts as
  JSON literals (`json.dumps(...)`), which Python parses as safe string
  literals. Never use plain `.format()` for user data here — it opens code
  injection via embedded quotes.
- All tools call `executor._resolve_project_dir()` to enforce the work-dir
  sandbox before running commands.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP


def _py_literal(value) -> str:
    """Return a Python source-level literal for the given string/obj.

    ``json.dumps`` output is a valid Python string literal for str inputs
    (quoting and escaping rules match). NEVER interpolate user strings into
    Python source via ``.format()`` or f-strings carrying user data.
    """
    return json.dumps(value)


def _resolve_input_path(executor, path: str) -> Path | None:
    """Resolve optional user-supplied file paths under the configured work_dir."""
    if not path:
        return None
    return executor._resolve_project_dir(path)


def register_ai_tools(mcp: FastMCP, executor):
    """Register 11 AI inference tools (9 adapted from ATAC + 2 RNA-specific)."""
    q = executor.quote
    PY = os.environ.get("RNA_MCP_PYTHON", "/home/ai/miniconda3/bin/python3")

    def _safe_dir(project_dir: str) -> str:
        return str(executor._resolve_project_dir(project_dir))

    # ── 1. Sample Auto-Detection ────────────────────────────────
    @mcp.tool()
    def detect_samples(project_dir: str) -> str:
        """Scan directory for FASTQ files and auto-detect sample list with read structure.
        Recognises Illumina, BGI, MGI, SRA naming conventions (RNA: _1/_2 common).

        Args:
            project_dir: Directory to scan for FASTQ files.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            "import glob, os, json, re\n"
            "fqs = sorted(glob.glob(os.path.join(PROJECT_DIR, '*.fastq.gz')) +\n"
            "             glob.glob(os.path.join(PROJECT_DIR, '*.fq.gz')))\n"
            "samples = {}\n"
            "for fq in fqs:\n"
            "    base = os.path.basename(fq)\n"
            "    for pat in [\n"
            "        r'^(.+?)_[12]\\.f',\n"
            "        r'^(.+?)_R?[12][\\._]',\n"
            "        r'^(SRR\\d+)[\\._]',\n"
            "        r'^(.+?)\\.f',\n"
            "    ]:\n"
            "        m = re.match(pat, base)\n"
            "        if m:\n"
            "            samples.setdefault(m.group(1), []).append(base)\n"
            "            break\n"
            "result = []\n"
            "for name, files in sorted(samples.items()):\n"
            "    paired = any('_2.' in f or '_R2' in f for f in files)\n"
            "    result.append({'sample': name, 'files': files, 'paired_end': paired})\n"
            "print(json.dumps(result, indent=2))\n"
        )
        return executor.run(f"{PY} -c {q(script)}", cwd=d)

    # ── 2. Read Format Auto-Detection ───────────────────────────
    @mcp.tool()
    def detect_read_format(project_dir: str, sample: str) -> str:
        """Detect sequencing platform and read length from FASTQ header.

        Args:
            project_dir: Project directory.
            sample:      Sample prefix.
        """
        d = _safe_dir(project_dir)
        # All user strings are embedded as Python literals to prevent shell
        # and Python code injection.
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            f"SAMPLE = {_py_literal(sample)}\n"
            "import glob, gzip, json, os, re, sys\n"
            "cands = sorted(\n"
            "    glob.glob(os.path.join(PROJECT_DIR, SAMPLE + '_1.fastq.gz')) +\n"
            "    glob.glob(os.path.join(PROJECT_DIR, SAMPLE + '*R1*.fastq.gz'))\n"
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
        return executor.run(f"{PY} -c {q(script)}", cwd=d)

    # ── 3. Treatment/Control Group Classification ───────────────
    @mcp.tool()
    def classify_groups(project_dir: str) -> str:
        """Auto-classify samples into treatment/control from filename keywords.
        RNA-seq-aware keywords: OE, KO, KD, shRNA, siRNA, treated, drug, stimul;
        Control: WT, ctrl, vehicle, DMSO, scramble, mock, EV, empty, DDK.

        Args:
            project_dir: Project directory.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            "import glob, os, json, re\n"
            "fqs = glob.glob(os.path.join(PROJECT_DIR, '*_1.fastq.gz')) + \\\n"
            "      glob.glob(os.path.join(PROJECT_DIR, '*_R1.fastq.gz'))\n"
            "samples = [os.path.basename(f).split('_1.fastq')[0].split('_R1')[0] for f in fqs]\n"
            "tkw = re.compile(r'(OE|KO|KD|shRNA|siRNA|(?<!un)treated|drug|inhibitor|stimul)', re.I)\n"
            "ckw = re.compile(r'(WT|wt|ctrl|control|DDK|vehicle|DMSO|scramble|mock|untreated|EV|empty)', re.I)\n"
            "groups = {'treatment': [], 'control': [], 'unclassified': []}\n"
            "for s in sorted(set(samples)):\n"
            "    if tkw.search(s):   groups['treatment'].append(s)\n"
            "    elif ckw.search(s): groups['control'].append(s)\n"
            "    else:               groups['unclassified'].append(s)\n"
            "print(json.dumps(groups, indent=2))\n"
        )
        return executor.run(f"{PY} -c {q(script)}", cwd=d)

    # ── 4. Genome/Species Auto-Detection ────────────────────────
    @mcp.tool()
    def detect_genome(project_dir: str) -> str:
        """Infer genome build from directory/file names.

        Args:
            project_dir: Project directory.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            "import os, json, re\n"
            "pats = {\n"
            "    'hg38': r'(hg38|GRCh38)',\n"
            "    'hg19': r'(hg19|GRCh37)',\n"
            "    'mm10': r'(mm10|GRCm38)',\n"
            "    'mm39': r'(mm39|GRCm39)',\n"
            "    'rn6':  r'(rn6|Rnor_6)',\n"
            "    'dm6':  r'(dm6|BDGP6)',\n"
            "}\n"
            "found = {}\n"
            "for root, dirs, files in os.walk(PROJECT_DIR):\n"
            "    for name in dirs + files:\n"
            "        full = os.path.join(root, name)\n"
            "        for gn, pat in pats.items():\n"
            "            if re.search(pat, full):\n"
            "                found[gn] = found.get(gn, 0) + 1\n"
            "if found:\n"
            "    best = max(found, key=found.get)\n"
            "    print(json.dumps({'detected': best, 'confidence': 'high' if found[best]>2 else 'low', 'all_hits': found}))\n"
            "else:\n"
            "    print(json.dumps({'detected': None, 'message': 'Could not auto-detect genome.'}))\n"
        )
        return executor.run(f"{PY} -c {q(script)}", cwd=d)

    # ── 5. DE Contrast/Design Auto-Inference ────────────────────
    @mcp.tool()
    def suggest_de_contrast(project_dir: str) -> str:
        """Infer DE contrast + mapfile skeleton from sample names.

        Strategies (in order):
        1) Keyword match (OE/KO/KD/... vs WT/ctrl/DMSO/...) — high confidence.
        2) Fallback: split samples in half (first half untreated, second half
           treated) as a tentative proposal — low confidence, marked as
           `needs_user_confirmation`.

        Output fields:
        - ``status``: 'ready' | 'needs_user_confirmation' | 'no_samples'
        - ``strategy``: which heuristic produced the grouping
        - ``mapfile_preview``: TSV ready to write to mapfile.txt
        - ``user_message``: a plain-language sentence suitable to show the
          user for confirmation before running DE.

        Args:
            project_dir: Project directory.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            "import glob, os, json, re\n"
            "cands = glob.glob(os.path.join(PROJECT_DIR, '*.genes.results'))\n"
            "if not cands:\n"
            "    cands = (glob.glob(os.path.join(PROJECT_DIR, '*_1.fastq.gz')) +\n"
            "             glob.glob(os.path.join(PROJECT_DIR, '*_R1.fastq.gz')) +\n"
            "             glob.glob(os.path.join(PROJECT_DIR, '*_R1_001.fastq.gz')))\n"
            "def _strip(n):\n"
            "    n = os.path.basename(n)\n"
            "    for suf in ('.genes.results', '_R1_001.fastq.gz', '_R1.fastq.gz',\n"
            "                '_1.fastq.gz', '.fastq.gz', '.fq.gz'):\n"
            "        if n.endswith(suf): return n[:-len(suf)]\n"
            "    return n\n"
            "names = sorted({_strip(c) for c in cands})\n"
            "tkw = re.compile(r'(OE|KO|KD|shRNA|siRNA|(?<!un)treated|drug|inhibitor|stimul)', re.I)\n"
            "ckw = re.compile(r'(WT|wt|ctrl|control|DDK|vehicle|DMSO|scramble|mock|untreated|EV|empty)', re.I)\n"
            "rows = []; reasons = {}\n"
            "for n in names:\n"
            "    tm = tkw.search(n); cm = ckw.search(n)\n"
            "    if tm and not cm:\n"
            "        rows.append((n, 'treated')); reasons[n] = f'keyword: {tm.group(0)}'\n"
            "    elif cm and not tm:\n"
            "        rows.append((n, 'untreated')); reasons[n] = f'keyword: {cm.group(0)}'\n"
            "    else:\n"
            "        rows.append((n, '?')); reasons[n] = 'no keyword match'\n"
            "t_n = sum(1 for _, c in rows if c == 'treated')\n"
            "c_n = sum(1 for _, c in rows if c == 'untreated')\n"
            "u_n = sum(1 for _, c in rows if c == '?')\n"
            "strategy = 'keyword'\n"
            "user_message = ''\n"
            "status = 'ready' if (t_n and c_n and u_n == 0) else None\n"
            "if not names:\n"
            "    status = 'no_samples'\n"
            "    strategy = 'none'\n"
            "    user_message = 'No RSEM results or FASTQ files found in project_dir.'\n"
            "elif u_n == len(rows):\n"
            "    # All samples unclassified — do NOT fabricate a grouping\n"
            "    status = 'needs_user_confirmation'\n"
            "    strategy = 'none'\n"
            "    user_message = ('Could not infer groups from sample names. '\n"
            "                    'Fill in the TREATMENT column (treated/untreated) '\n"
            "                    'in mapfile_preview manually before running step4/5.')\n"
            "elif u_n > 0:\n"
            "    status = 'needs_user_confirmation'\n"
            "    strategy = 'keyword_partial'\n"
            "    unclassified = [n for n, c in rows if c == '?']\n"
            "    user_message = (\n"
            "        f'Matched {t_n} treated / {c_n} untreated by keyword, but '\n"
            "        f'{u_n} sample(s) unclassified: {unclassified}. '\n"
            "        'Please edit mapfile_preview to set their TREATMENT value.'\n"
            "    )\n"
            "tsv = 'sample\\tTREATMENT\\n' + '\\n'.join(f'{s}\\t{c}' for s, c in rows)\n"
            "rec = {\n"
            "    'status':          status or 'ready',\n"
            "    'strategy':        strategy,\n"
            "    'design_formula':  '~TREATMENT',\n"
            "    'contrast':        ['TREATMENT', 'treated', 'untreated'],\n"
            "    'reference':       'untreated',\n"
            "    'counts':          {'treated': t_n, 'untreated': c_n, 'unknown': u_n},\n"
            "    'per_sample_reason': reasons,\n"
            "    'mapfile_preview': tsv,\n"
            "    'user_message':    user_message or f'Matched {t_n} treated / {c_n} untreated by keyword.',\n"
            "}\n"
            "print(json.dumps(rec, indent=2))\n"
        )
        return executor.run(f"{PY} -c {q(script)}", cwd=d)

    # ── 6. STAR Options Auto-Recommendation ─────────────────────
    @mcp.tool()
    def recommend_star_options(project_dir: str, sample: str) -> str:
        """Analyse FASTQ header/read length to recommend STAR parameters:
        --sjdbOverhang (= read_len - 1), --readFilesCommand, etc.

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
            "    glob.glob(os.path.join(PROJECT_DIR, SAMPLE + '_1.fastq.gz')) +\n"
            "    glob.glob(os.path.join(PROJECT_DIR, SAMPLE + '*R1*.fastq.gz'))\n"
            ")\n"
            "if not cands:\n"
            "    print(json.dumps({'error': 'R1 not found', 'sample': SAMPLE}))\n"
            "    sys.exit(1)\n"
            "r1 = cands[0]\n"
            "read_cmd = 'zcat' if r1.endswith('.gz') else '-'\n"
            "with gzip.open(r1, 'rt') as fh:\n"
            "    fh.readline()\n"
            "    seq = fh.readline().rstrip('\\n')\n"
            "rl = len(seq)\n"
            "if rl <= 50:\n"
            "    mm, note = 5, 'Short reads (<=50bp): stricter mismatch filter'\n"
            "elif rl <= 100:\n"
            "    mm, note = 10, 'Standard reads (50-100bp)'\n"
            "else:\n"
            "    mm, note = 15, 'Long reads (>100bp): loose mismatch filter'\n"
            "print(json.dumps({\n"
            "    'read_length': rl,\n"
            "    'recommendations': {\n"
            "        'readFilesCommand': read_cmd,\n"
            "        'sjdbOverhang': rl - 1,\n"
            "        'quantMode': 'TranscriptomeSAM',\n"
            "        'outFilterMismatchNmax': mm,\n"
            "        'note': note,\n"
            "    },\n"
            "}, indent=2))\n"
        )
        return executor.run(f"{PY} -c {q(script)}", cwd=d)

    # ── 7. QC Auto-Judgment (STAR + RSEM + FastQC) ──────────────
    @mcp.tool()
    def check_qc(project_dir: str, sample: str) -> str:
        """Parse STAR Log.final.out, RSEM .cnt, FastQC results. PASS/WARN/FAIL per metric.

        Args:
            project_dir: Project directory.
            sample:      Sample prefix.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            f"SAMPLE = {_py_literal(sample)}\n"
            "import os, json, re\n"
            "res = {'sample': SAMPLE, 'metrics': {}, 'verdict': 'PASS'}\n"
            "def mark(status):\n"
            "    order = {'PASS': 0, 'WARN': 1, 'FAIL': 2}\n"
            "    if order[status] > order[res['verdict']]: res['verdict'] = status\n"
            "# STAR Log.final.out\n"
            "log = os.path.join(PROJECT_DIR, SAMPLE + 'Log.final.out')\n"
            "if os.path.exists(log):\n"
            "    t = open(log).read()\n"
            "    for key, pat in [\n"
            "        ('uniquely_mapped_pct', r'Uniquely mapped reads %\\s*\\|\\s*([\\d.]+)%'),\n"
            "        ('multi_mapped_pct',   r'% of reads mapped to multiple loci\\s*\\|\\s*([\\d.]+)%'),\n"
            "        ('unmapped_pct',       r'% of reads unmapped: too short\\s*\\|\\s*([\\d.]+)%'),\n"
            "    ]:\n"
            "        m = re.search(pat, t)\n"
            "        if m:\n"
            "            v = float(m.group(1))\n"
            "            status = 'PASS'\n"
            "            if key == 'uniquely_mapped_pct':\n"
            "                status = 'PASS' if v >= 70 else ('WARN' if v >= 50 else 'FAIL')\n"
            "            elif key == 'unmapped_pct':\n"
            "                status = 'PASS' if v <= 10 else ('WARN' if v <= 30 else 'FAIL')\n"
            "            res['metrics'][key] = {'value': v, 'status': status}\n"
            "            mark(status)\n"
            "# RSEM .stat/SAMPLE.cnt\n"
            "cnt = os.path.join(PROJECT_DIR, SAMPLE + '.stat', SAMPLE + '.cnt')\n"
            "if os.path.exists(cnt):\n"
            "    with open(cnt) as f:\n"
            "        line1 = f.readline().strip().split()\n"
            "    try:\n"
            "        N0, N1, N2, N_tot = [int(x) for x in line1[:4]]\n"
            "        assigned_pct = 100.0 * N1 / N_tot if N_tot else 0\n"
            "        status = 'PASS' if assigned_pct >= 60 else ('WARN' if assigned_pct >= 40 else 'FAIL')\n"
            "        res['metrics']['rsem_assigned_pct'] = {'value': round(assigned_pct, 2), 'status': status}\n"
            "        mark(status)\n"
            "    except Exception as e:\n"
            "        res['metrics']['rsem_error'] = str(e)\n"
            "print(json.dumps(res, indent=2))\n"
        )
        return executor.run(f"{PY} -c {q(script)}", cwd=d)

    # ── 8. Resume/Checkpoint Status ─────────────────────────────
    @mcp.tool()
    def checkpoint_status(project_dir: str) -> str:
        """Show pipeline checkpoint for all samples.

        Args:
            project_dir: Project directory.
        """
        d = _safe_dir(project_dir)
        data = executor.load_checkpoint(d)
        all_steps = ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8"]
        summary = {}
        for sample, done in data.get("completed_steps", {}).items():
            summary[sample] = {
                "completed": done,
                "remaining": [s for s in all_steps if s not in done],
            }
        return json.dumps({"checkpoint": summary}, indent=2)

    # ── 9. Resource Auto-Configuration ──────────────────────────
    @mcp.tool()
    def detect_resources() -> str:
        """Detect CPU cores, RAM, GPU. Recommend num_threads and STAR --limitGenomeGenerateRAM."""
        r = executor.detect_resources()
        cores, mem = r["cores"], r["mem_gb"]
        rec_t = max(1, cores - 2)
        star_ram = max(30, mem * 3 // 4)
        gpu = executor.run(
            "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo 'No GPU'"
        )
        return json.dumps({
            "cpu_cores": cores,
            "available_ram_gb": mem,
            "recommended_threads": rec_t,
            "star_limitGenomeGenerateRAM": f"{star_ram*1024**3}",
            "gpu": gpu.strip(),
        }, indent=2)

    # ══════════════════════════════════════════════════════════════
    # RNA-specific additions (tools #10, #11)
    # ══════════════════════════════════════════════════════════════

    # ── 10. Strandedness Auto-Detection ─────────────────────────
    @mcp.tool()
    def detect_strandedness(project_dir: str, sample: str) -> str:
        """Detect library strandedness from RSEM .stat/.model file (forward_prob).
        Returns 'unstranded' / 'forward' / 'reverse' + confidence + recommended
        --forward-prob value (1.0 / 0.0 / 0.5) for RSEM rerun if needed.

        Args:
            project_dir: Project directory (with {sample}.stat/ subdirectory).
            sample:      Sample prefix.
        """
        d = _safe_dir(project_dir)
        script = (
            f"PROJECT_DIR = {_py_literal(d)}\n"
            f"SAMPLE = {_py_literal(sample)}\n"
            "import os, json, re\n"
            "stat = os.path.join(PROJECT_DIR, SAMPLE + '.stat')\n"
            "model = os.path.join(stat, SAMPLE + '.model')\n"
            "out = {'sample': SAMPLE, 'method': None}\n"
            "\n"
            "if not os.path.isdir(stat):\n"
            "    out['error'] = f'RSEM stat directory not found: {stat}. Run step3 first.'\n"
            "elif not os.path.exists(model):\n"
            "    out['error'] = f'RSEM model file not found: {model}'\n"
            "else:\n"
            "    # RSEM .model: line1=model_type, line3=forward_prob (blank-separated)\n"
            "    with open(model) as f:\n"
            "        lines = [ln.strip() for ln in f.readlines()[:10]]\n"
            "    numeric = [ln for ln in lines if ln and re.match(r'^-?[0-9.]+\\s*$', ln)]\n"
            "    fp = None\n"
            "    if len(numeric) >= 2:\n"
            "        try: fp = float(numeric[1])\n"
            "        except ValueError: pass\n"
            "    if fp is None:\n"
            "        out['error'] = f'Could not parse forward_prob from {model}'\n"
            "    else:\n"
            "        out['forward_prob'] = fp\n"
            "        out['method'] = 'RSEM .model'\n"
            "        if 0.8 <= fp <= 1.0:\n"
            "            out['strandedness'] = 'forward'\n"
            "            out['rsem_flag']    = '--forward-prob 1.0'\n"
            "            out['confidence']   = 'high'\n"
            "        elif 0.0 <= fp <= 0.2:\n"
            "            out['strandedness'] = 'reverse'\n"
            "            out['rsem_flag']    = '--forward-prob 0.0'\n"
            "            out['confidence']   = 'high'\n"
            "        elif 0.4 <= fp <= 0.6:\n"
            "            out['strandedness'] = 'unstranded'\n"
            "            out['rsem_flag']    = '--forward-prob 0.5'\n"
            "            out['confidence']   = 'high'\n"
            "        else:\n"
            "            out['strandedness'] = 'ambiguous'\n"
            "            out['rsem_flag']    = '--forward-prob 0.5'\n"
            "            out['confidence']   = 'low'\n"
            "print(json.dumps(out, indent=2))\n"
        )
        return executor.run(f"{PY} -c {q(script)}", cwd=d)

    # ── 11. Gene ID → Symbol Annotation ─────────────────────────
    @mcp.tool()
    def annotate_symbols(
        project_dir: str,
        results_file: str = "",
        gtf_file: str = "",
        out_name: str = "DESeq2_results_annotated.csv",
    ) -> str:
        """Map Ensembl gene IDs in DE results to gene symbols using a GTF.
        Looks up 'gene_id' → 'gene_name' attributes in the GTF.

        Args:
            project_dir:  Project directory.
            results_file: DE results CSV. Default: project_dir/DESeq2_results.csv.
            gtf_file:     GTF path. Default tries {project_dir}/Reference/hg38/*.gtf.
            out_name:     Output filename (written into project_dir).
        """
        pd_dir = executor._resolve_project_dir(project_dir)
        results_path = _resolve_input_path(executor, results_file) or pd_dir / "DESeq2_results.csv"
        if not results_path.exists():
            return json.dumps({"error": f"Results not found: {results_path}"})

        gtf_path = _resolve_input_path(executor, gtf_file)
        if gtf_path is None:
            cand = list((pd_dir / "Reference" / "hg38").glob("*.gtf")) + \
                   list(pd_dir.glob("*.gtf"))
            if not cand:
                return json.dumps({
                    "error": "GTF not specified and none found under project_dir. "
                             "Provide gtf_file argument."
                })
            gtf_path = cand[0]

        # Parse GTF attributes column (9th, tab-separated).
        # Store both versioned (ENSG00000000003.15) and unversioned
        # (ENSG00000000003) keys so GENCODE-style IDs match regardless of
        # whether the DE results carry a version suffix or not.
        id_to_name: dict[str, str] = {}
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#") or "\tgene\t" not in line:
                    continue
                attr = line.rstrip().split("\t")[8]
                gid = _gtf_attr(attr, "gene_id")
                gname = _gtf_attr(attr, "gene_name") or gid
                if gid:
                    id_to_name[gid] = gname
                    base = gid.split(".")[0]
                    if base not in id_to_name:
                        id_to_name[base] = gname

        res = pd.read_csv(results_path, index_col=0)
        res.insert(0, "gene_symbol", [
            id_to_name.get(g) or id_to_name.get(g.split(".")[0], g)
            for g in res.index
        ])
        out_path = pd_dir / out_name
        res.to_csv(out_path)
        mapped = sum(1 for g in res.index if g in id_to_name)
        return json.dumps({
            "output": str(out_path),
            "total_genes": len(res),
            "mapped_to_symbol": mapped,
            "gtf_used": str(gtf_path),
        }, indent=2)


def _gtf_attr(attr: str, key: str) -> str | None:
    """Extract a value like gene_id "ENSG..."; from a GTF attributes string."""
    needle = f'{key} "'
    i = attr.find(needle)
    if i < 0:
        return None
    i += len(needle)
    j = attr.find('"', i)
    return attr[i:j] if j > i else None
