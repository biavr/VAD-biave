"""Live web dashboard for an in-progress VAD training run.

Reads directly from the same vis_data/scalars.json file mmengine's own
LocalVisBackend already writes to during training -- no config changes, no
training-pipeline hook, no new dependency. Point it at a config's work_dir
and it serves a live-updating page of loss curves, picking up the latest
timestamped run folder automatically (so relaunching training under the
same work_dir doesn't require restarting this).

Usage (run in a separate terminal/pane from your training command):
    python tools/live_dashboard/server.py --work-dir /workspace/logs/outputs_tiny_stage_1

Then open the printed http://localhost:<port> URL in a browser (on a remote
workstation accessed through VS Code, its Ports panel will offer to forward
it automatically). See README.md in this folder for more detail.
"""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / 'template.html'
RUN_DIR_RE = re.compile(r'^\d{8}_\d{6}$')
CKPT_RE = re.compile(r'epoch_(\d+)\.pth$')
POLL_MS = 3000


def find_latest_run(work_dir: Path):
    if not work_dir.is_dir():
        return None
    candidates = [d for d in work_dir.iterdir() if d.is_dir() and RUN_DIR_RE.match(d.name)]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.name)


def load_scalars(run_dir: Path):
    path = run_dir / 'vis_data' / 'scalars.json'
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Training writes this file one line at a time; the very
                # last line can be mid-write when we read it. Skip it this
                # poll, it'll be complete (and included) on the next one.
                continue
    return records


def find_checkpoints(work_dir: Path):
    epochs = set()
    if work_dir.is_dir():
        for f in work_dir.rglob('epoch_*.pth'):
            m = CKPT_RE.search(f.name)
            if m:
                epochs.add(int(m.group(1)))
    return sorted(epochs)


OCC_PREVIEW_NAME_RE = re.compile(r'^sample_\d+_latest\.png$')


def find_occ_previews(work_dir: Path):
    """Latest occupancy-prediction preview per tracked sample, written by
    occ_watcher.py. Returns [] if that watcher isn't running (or hasn't
    produced a first render yet) -- this is optional, not required for the
    rest of the dashboard to work."""
    preview_dir = work_dir / 'occ_preview'
    if not preview_dir.is_dir():
        return []
    return sorted(f.name for f in preview_dir.iterdir() if OCC_PREVIEW_NAME_RE.match(f.name))


def make_handler(work_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self._send(200, 'text/html; charset=utf-8', TEMPLATE_PATH.read_bytes())
                return
            if self.path.startswith('/data'):
                run_dir = find_latest_run(work_dir)
                payload = {
                    'run': run_dir.name if run_dir else None,
                    'work_dir': str(work_dir),
                    'records': load_scalars(run_dir) if run_dir else [],
                    'checkpoints': find_checkpoints(work_dir),
                    'occ_previews': find_occ_previews(work_dir),
                    'poll_ms': POLL_MS,
                    'scalars_path': str((run_dir / 'vis_data' / 'scalars.json')) if run_dir else None,
                }
                self._send(200, 'application/json', json.dumps(payload).encode())
                return
            if self.path.startswith('/occ_preview/'):
                name = self.path[len('/occ_preview/'):].split('?', 1)[0]
                if not OCC_PREVIEW_NAME_RE.match(name):
                    self._send(404, 'text/plain', b'not found')
                    return
                img_path = work_dir / 'occ_preview' / name
                if not img_path.is_file():
                    self._send(404, 'text/plain', b'not found')
                    return
                self._send(200, 'image/png', img_path.read_bytes())
                return
            self._send(404, 'text/plain', b'not found')

        def log_message(self, fmt, *args):
            pass  # keep the terminal quiet; errors still raise normally

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work-dir', required=True, help="A config's work_dir, e.g. /workspace/logs/outputs_tiny_stage_1")
    parser.add_argument('--port', type=int, default=8787)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    server = ThreadingHTTPServer(('0.0.0.0', args.port), make_handler(work_dir))
    print(f'Watching {work_dir}')
    print(f'Live dashboard: http://localhost:{args.port}')
    print('Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
