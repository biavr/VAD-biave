"""One-shot health check for an in-progress VAD training run.

Reuses the same run-discovery and scalars.json parsing as server.py. Prints
a single status line and exits 0 (healthy) or 1 (needs attention), so it's
easy to call from a periodic check: a non-zero exit, or a status other than
OK, means something needs a human to look.

Usage:
    python tools/live_dashboard/healthcheck.py --work-dir /workspace/logs/outputs_tiny_stage_1

Detects:
    STALE          - scalars.json hasn't been written to in --stale-secs
                      (default 600s = 10min). This is the signal that would
                      have caught the NCCL watchdog hang we hit before, long
                      before its 30-minute timeout fired.
    DIVERGED       - latest logged loss or grad_norm is NaN/Inf.
    PROCESS_EXITED - no process matching train_new.py is running anymore,
                      whether from a crash or a normal finish; the log tail
                      is included so a human can tell which at a glance.
    OK             - none of the above.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from server import find_latest_run, load_scalars, RUN_DIR_RE  # noqa: E402

TRAIN_CMD_HINT = 'train_new.py'
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../VAD
WORK_DIR_RE = re.compile(r"""work_dir\s*=\s*['"]([^'"]+)['"]""")


def _config_work_dir(config_path: Path):
    """Read a config file's own work_dir = '...' line without needing a full
    mmengine Config parse (this just needs to be fast and read-only)."""
    try:
        text = config_path.read_text(errors='replace')
    except (FileNotFoundError, PermissionError):
        return None
    m = WORK_DIR_RE.search(text)
    return Path(m.group(1)).resolve() if m else None


def run_start_time(run_dir: Path):
    """Parse the wall-clock start time encoded in mmengine's run folder name
    (YYYYMMDD_HHMMSS)."""
    if not RUN_DIR_RE.match(run_dir.name):
        return None
    return datetime.strptime(run_dir.name, '%Y%m%d_%H%M%S')


def _process_start_time(pid: str):
    try:
        out = subprocess.run(['ps', '-o', 'lstart=', '-p', pid], capture_output=True, text=True, timeout=5)
        return datetime.strptime(out.stdout.strip(), '%a %b %d %H:%M:%S %Y')
    except Exception:
        return None  # process vanished mid-check, or parsing failed


def is_training_process_alive(work_dir: Path, not_before: datetime):
    """True if some running process is training with a config whose own
    work_dir resolves to this exact directory, AND that process started at
    or after this specific run's own start time.

    Neither signal alone is enough. A plain substring match on
    'train_new.py' catches orphaned processes from any unrelated run. Config
    path alone isn't enough either: an orphaned process from a previous,
    now-dead launch of the *same* config has an identical cmdline to a
    currently-running one -- only comparing against this run folder's own
    start timestamp tells them apart.
    """
    proc = Path('/proc')
    if not proc.is_dir():
        return None  # not on Linux / no /proc -- can't tell, don't claim either way
    work_dir = work_dir.resolve()
    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            raw = (pid_dir / 'cmdline').read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        argv = raw.decode(errors='replace').split('\x00')
        if not any(TRAIN_CMD_HINT in a for a in argv):
            continue
        config_args = [a for a in argv if a.endswith('.py') and TRAIN_CMD_HINT not in a]
        if not any(_config_work_dir((REPO_ROOT / a) if not os.path.isabs(a) else Path(a)) == work_dir
                  for a in config_args):
            continue
        started = _process_start_time(pid_dir.name)
        if started is not None and (not_before is None or started >= not_before):
            return True
    return False


LOG_LINE_TS_RE = re.compile(r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})')


def _log_lines(run_dir: Path):
    log_files = list(run_dir.glob('*.log'))
    if not log_files:
        return []
    return log_files[0].read_text(errors='replace').splitlines()


def log_tail(run_dir: Path, n_lines=25):
    return '\n'.join(_log_lines(run_dir)[-n_lines:])


def last_log_timestamp(run_dir: Path):
    """Wall-clock time of the most recent *content* written to this run's
    .log file, parsed from mmengine's own per-line timestamp.

    Deliberately not filesystem mtime: on this NFS-backed work_dir, mtime
    has been observed to advance without any new line actually being
    appended (likely attribute-cache refresh noise), which let a real
    multi-hour hang read back as fresh/OK. Line content is the only signal
    that can't lie about whether training actually produced new output.
    """
    for line in reversed(_log_lines(run_dir)):
        m = LOG_LINE_TS_RE.match(line)
        if m:
            return datetime.strptime(m.group(1), '%Y/%m/%d %H:%M:%S')
    return None


def check(work_dir: Path, stale_secs: int):
    run_dir = find_latest_run(work_dir)
    if run_dir is None:
        return 'NO_RUN', f'No timestamped run folder found yet under {work_dir}', {}

    records = load_scalars(run_dir)
    # torchrun launches a few seconds (occasionally longer, under load) before
    # mmengine creates this run folder, so give the start-time comparison a
    # buffer rather than a hard cutoff at the folder's own timestamp.
    run_start = run_start_time(run_dir)
    not_before = (run_start - timedelta(minutes=5)) if run_start else None
    alive = is_training_process_alive(work_dir, not_before)

    if records:
        last = records[-1]
        if math.isnan(last.get('loss', 0)) or math.isinf(last.get('loss', 0)) or \
           math.isnan(last.get('grad_norm', 0)) or math.isinf(last.get('grad_norm', 0)):
            return 'DIVERGED', (
                f"loss/grad_norm is NaN or Inf at epoch {last.get('epoch')} step {last.get('step')} "
                f"(run {run_dir.name})"
            ), {'record': last}

    last_ts = last_log_timestamp(run_dir)
    if last_ts is not None:
        age = (datetime.now() - last_ts).total_seconds()
        if age > stale_secs:
            if alive is False:
                return 'PROCESS_EXITED', (
                    f"training process is gone and the log has been quiet for {int(age)}s "
                    f"(run {run_dir.name})"
                ), {'log_tail': log_tail(run_dir)}
            return 'STALE', (
                f"no new logged iteration in {int(age)}s (threshold {stale_secs}s), "
                f"but the process is still alive -- this is exactly the NCCL-hang signature "
                f"(run {run_dir.name})"
            ), {'log_tail': log_tail(run_dir)}

    if alive is False and not records:
        return 'PROCESS_EXITED', f'No training process found and no iterations logged yet (run {run_dir.name})', {
            'log_tail': log_tail(run_dir)
        }

    last = records[-1] if records else {}
    return 'OK', (
        f"run {run_dir.name}: epoch {last.get('epoch','?')} step {last.get('step','?')} "
        f"loss {last.get('loss','?')}"
    ), {'record': last}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--stale-secs', type=int, default=600)
    args = parser.parse_args()

    status, message, extra = check(Path(args.work_dir), args.stale_secs)
    print(f'{status}: {message}')
    if extra.get('log_tail'):
        print('--- log tail ---')
        print(extra['log_tail'])
    sys.exit(0 if status == 'OK' else 1)


if __name__ == '__main__':
    main()
