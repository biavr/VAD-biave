#!/usr/bin/env bash
# One-time setup for the nuscenes_rerun visualization plugin.
#
# This does NOT install, upgrade, or touch any Python package -- rerun-sdk
# must already be pip-installed. It only:
#   1. Makes `rerun-importer-nuscenes` executable.
#   2. Symlinks it into ~/.local/bin (the same directory `pip install --user`
#      already puts the `rerun` CLI binary in).
#   3. Appends ~/.local/bin to $PATH via ~/.bashrc, if it isn't there yet.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BIN="$HOME/.local/bin"

mkdir -p "$LOCAL_BIN"
chmod +x "$HERE/rerun-importer-nuscenes"
ln -sf "$HERE/rerun-importer-nuscenes" "$LOCAL_BIN/rerun-importer-nuscenes"
echo "Linked $LOCAL_BIN/rerun-importer-nuscenes -> $HERE/rerun-importer-nuscenes"

PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if echo "$PATH" | tr ':' '\n' | grep -qx "$LOCAL_BIN"; then
    echo "$LOCAL_BIN is already on \$PATH"
elif grep -qxF "$PATH_LINE" "$HOME/.bashrc" 2>/dev/null; then
    echo "$LOCAL_BIN PATH export already present in ~/.bashrc (open a new shell to pick it up)"
else
    echo "$PATH_LINE" >> "$HOME/.bashrc"
    echo "Added $LOCAL_BIN to \$PATH in ~/.bashrc -- run 'source ~/.bashrc' or open a new shell"
fi

cat <<'EOF'

Try it out with:
  rerun /workspace/datasets/nuscenes/lidarseg/v1.0-trainval/0a0c9ff1674645fdab2cf6d7308b9269_lidarseg.bin

Or without touching $PATH at all, via the CLI directly:
  python3 -m nuscenes_rerun lidarseg <path_to_lidarseg.bin>
EOF
