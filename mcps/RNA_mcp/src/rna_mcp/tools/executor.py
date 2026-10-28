"""Command executor — runs shell/python commands locally on the HPC."""
import subprocess
import signal
import os
import json
import shlex
import fcntl
import pathlib
from pathlib import Path




_AGENT_BIN = "/BIO1/MRAG/Agent/bin/tools/bin"
_EXTRA_PATHS = [_AGENT_BIN, "/usr/local/bin", "/usr/local/sbin"]
def _agent_env():
    env = os.environ.copy()
    parts = env.get("PATH", "").split(":") if env.get("PATH") else []
    for p in reversed(_EXTRA_PATHS):
        if p not in parts:
            parts.insert(0, p)
    env["PATH"] = ":".join(parts)
    return env

class LocalExecutor:
    """Execute commands directly on the HPC filesystem."""

    def __init__(self, work_dir: str = "/BIO1/MRAG"):
        self.work_dir = work_dir

    @staticmethod
    def quote(s: str) -> str:
        """Shell-escape a string to prevent injection."""
        return shlex.quote(s)

    @staticmethod
    def is_success(output: str) -> bool:
        """Check run() output for failure markers (non-zero exit, timeout, exception)."""
        if output.startswith(("[ERROR]", "[TIMEOUT]")):
            return False
        if "[EXIT CODE] " in output:
            return False
        return True

    # ── Path safety ─────────────────────────────────────────────
    def _resolve_project_dir(self, project_dir: str) -> Path:
        base = Path(self.work_dir).resolve()
        target = Path(project_dir or self.work_dir).resolve()
        if base not in (target, *target.parents):
            raise ValueError(f"Project dir must be inside {base}, got {target}")
        return target

    # ── Command execution ───────────────────────────────────────
    def run(self, cmd: str, cwd: str | None = None, timeout: int | None = None) -> str:
        wd = str(self._resolve_project_dir(cwd or self.work_dir))
        try:
            proc = subprocess.Popen(
                cmd, shell=True, executable="/bin/bash", cwd=wd, env=_agent_env(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                return f"[TIMEOUT] Command exceeded {timeout}s: {cmd[:100]}"
            output = stdout
            if stderr:
                output += "\n[STDERR]\n" + stderr
            if proc.returncode != 0:
                output += f"\n[EXIT CODE] {proc.returncode}"
            return output.strip()
        except Exception as e:
            return f"[ERROR] {e}"

    def run_bg(self, cmd: str, log_file: str, cwd: str | None = None) -> str:
        wd = str(self._resolve_project_dir(cwd or self.work_dir))
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as fh:
            subprocess.Popen(
                cmd, shell=True, executable="/bin/bash", cwd=wd, env=_agent_env(),
                stdout=fh, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return f"Background job started. Log: {log_path}"

    # ── Checkpoint helpers (file-lock protected) ────────────────
    def checkpoint_path(self, project_dir: str) -> Path:
        return self._resolve_project_dir(project_dir) / ".rna_checkpoint.json"

    def _lock_path(self, project_dir: str) -> Path:
        return self._resolve_project_dir(project_dir) / ".rna_checkpoint.lock"

    def load_checkpoint(self, project_dir: str) -> dict:
        cp = self.checkpoint_path(project_dir)
        if not cp.exists():
            return {"samples": {}, "completed_steps": {}}
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid checkpoint JSON at {cp}: {e}") from e
        data.setdefault("samples", {})
        data.setdefault("completed_steps", {})
        return data

    def save_checkpoint(self, project_dir: str, data: dict):
        cp = self.checkpoint_path(project_dir)
        tmp = cp.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(cp)

    def mark_done(self, project_dir: str, sample: str, step: str):
        lock_file = self._lock_path(project_dir)
        lock_file.touch(exist_ok=True)
        with open(lock_file, "r") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                data = self.load_checkpoint(project_dir)
                data["completed_steps"].setdefault(sample, [])
                if step not in data["completed_steps"][sample]:
                    data["completed_steps"][sample].append(step)
                self.save_checkpoint(project_dir, data)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _artifacts_present(self, project_dir: str, sample: str, step: str) -> bool:
        """Verify a step's real output artifacts exist on disk (size > 0).

        A checkpoint entry alone is NOT sufficient: outputs may have been
        deleted after the checkpoint was written (e.g. only FASTQ kept). Each
        step is only 'done' if its artifacts are physically present.
        """
        base = pathlib.Path(project_dir)
        if step == "step1":
            # PE trimmed pair OR SE trimmed single
            pe1 = base / "trim_galore" / f"{sample}_1_val_1.fq.gz"
            pe2 = base / "trim_galore" / f"{sample}_2_val_2.fq.gz"
            se  = base / "trim_galore" / f"{sample}_trimmed.fq.gz"
            if pe1.exists() and pe1.stat().st_size > 0 and pe2.exists() and pe2.stat().st_size > 0:
                return True
            return se.exists() and se.stat().st_size > 0
        if step == "step2":
            bam = base / f"{sample}Aligned.toTranscriptome.out.bam"
            log = base / f"{sample}Log.final.out"
            return (bam.exists() and bam.stat().st_size > 0
                    and log.exists() and log.stat().st_size > 0)
        if step == "step3":
            genes = base / f"{sample}.genes.results"
            return genes.exists() and genes.stat().st_size > 0
        # step4/step5 (global DESeq2/matrix) — trust checkpoint
        return True

    def is_done(self, project_dir: str, sample: str, step: str) -> bool:
        data = self.load_checkpoint(project_dir)
        if step not in data.get("completed_steps", {}).get(sample, []):
            return False
        return self._artifacts_present(project_dir, sample, step)

    # ── Resource detection ──────────────────────────────────────
    def detect_resources(self) -> dict:
        # usable CPUs (taskset / MRAG_CPU_LIST), not the host size (2026-09-16)
        try:
            cores = len(os.sched_getaffinity(0)) or 4
        except (AttributeError, OSError):
            cores = os.cpu_count() or 4
        try:
            mem_kb = int(self.run("awk '/MemAvailable/ {print $2}' /proc/meminfo").strip())
            mem_gb = mem_kb // 1024 // 1024
        except Exception:
            mem_gb = 8
        return {"cores": cores, "mem_gb": mem_gb}
