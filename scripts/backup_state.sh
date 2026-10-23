#!/usr/bin/env bash
# Consistent backup of the engine's runtime state (safe while services run).
#   sudo scripts/backup_state.sh [DEST_ROOT]        default DEST_ROOT=/var/backups/wallex
# SQLite files are copied with the online-backup API; everything else (JSONL logs, JSON state) is tarred.
# Override the state directories with STATE_DIRS="dir1 dir2". Keeps the newest $KEEP backups (default 14).
set -euo pipefail
DEST_ROOT="${1:-/var/backups/wallex}"
STATE_DIRS="${STATE_DIRS:-/var/lib/wallex-arena /var/lib/wallex-paper-agent}"
KEEP="${KEEP:-14}"
PYTHON="${PYTHON:-python3}"

dest="$DEST_ROOT/$(date -u +%Y%m%dT%H%M%SZ)"
install -d -m 0700 "$dest"
for dir in $STATE_DIRS; do
  [[ -d "$dir" ]] || { echo "skip missing $dir"; continue; }
  name="$(basename "$dir")"
  install -d -m 0700 "$dest/$name"
  tar -C "$(dirname "$dir")" --exclude='*.sqlite' --exclude='*.sqlite-wal' --exclude='*.sqlite-shm' \
      -czf "$dest/$name/files.tar.gz" "$name"
  find "$dir" -name '*.sqlite' -type f -print0 | while IFS= read -r -d '' db; do
    rel="${db#"$dir"/}"
    out="$dest/$name/sqlite/$rel"
    mkdir -p "$(dirname "$out")"
    "$PYTHON" - "$db" "$out" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
with dst:
    src.backup(dst)
src.close(); dst.close()
PY
  done
done
echo "backup written to $dest"
# Retention: keep the newest $KEEP timestamped directories.
ls -1d "$DEST_ROOT"/*/ 2>/dev/null | sort | head -n -"$KEEP" | xargs -r rm -rf
