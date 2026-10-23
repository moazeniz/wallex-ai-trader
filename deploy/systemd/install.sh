#!/usr/bin/env bash
# Render the systemd unit templates in this directory and (optionally) install them.
#
#   deploy/systemd/install.sh render            # render into ./build/systemd for review (no root needed)
#   sudo deploy/systemd/install.sh install      # render, create user/dirs, copy units, daemon-reload
#   sudo deploy/systemd/install.sh install --enable wallex-paper-agent wallex-arena
#
# It never installs the live drop-ins (deploy/systemd/live/*.conf) and never enables live trading.
# Override any setting through the environment, e.g. APP_DIR=/opt/wallex-trading SERVICE_USER=wallex.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

APP_DIR="${APP_DIR:-$REPO}"
VENV="${VENV:-$APP_DIR/.venv}"
SERVICE_USER="${SERVICE_USER:-wallex}"
SECRETS_DIR="${SECRETS_DIR:-/etc/wallex}"
ARENA_CONFIG="${ARENA_CONFIG:-$APP_DIR/experiments/wallex_arena/arena-config.json}"
WEB_DIR="${WEB_DIR:-/var/www/wallex-arena}"
DASHBOARD_PORT="${DASHBOARD_PORT:-9445}"
TLS_CERT="${TLS_CERT:-/etc/letsencrypt/live/<DASHBOARD_HOST>/fullchain.pem}"
TLS_KEY="${TLS_KEY:-/etc/letsencrypt/live/<DASHBOARD_HOST>/privkey.pem}"
OUT="${OUT:-$REPO/build/systemd}"
UNIT_DIR="${UNIT_DIR:-/etc/systemd/system}"

render() {
  mkdir -p "$OUT/live"
  local f
  for f in "$HERE"/*.service "$HERE"/*.timer "$HERE"/live/*.conf; do
    local rel="${f#"$HERE"/}"
    sed -e "s|@@APP_DIR@@|$APP_DIR|g" -e "s|@@VENV@@|$VENV|g" -e "s|@@SERVICE_USER@@|$SERVICE_USER|g" \
        -e "s|@@SECRETS_DIR@@|$SECRETS_DIR|g" -e "s|@@ARENA_CONFIG@@|$ARENA_CONFIG|g" -e "s|@@WEB_DIR@@|$WEB_DIR|g" \
        -e "s|@@DASHBOARD_PORT@@|$DASHBOARD_PORT|g" -e "s|@@TLS_CERT@@|$TLS_CERT|g" -e "s|@@TLS_KEY@@|$TLS_KEY|g" \
        "$f" > "$OUT/$rel"
  done
  if grep -rn '@@[A-Z_]*@@' "$OUT"; then echo "unrendered placeholder(s) above" >&2; exit 1; fi
  echo "Rendered units into $OUT"
}

install_units() {
  [[ $EUID -eq 0 ]] || { echo "install needs root (sudo)" >&2; exit 1; }
  case "$APP_DIR" in /home/*|/root/*) echo "APP_DIR under /home or /root is hidden by ProtectHome=yes; use e.g. /opt/wallex-trading" >&2; exit 1;; esac
  [[ -x "$VENV/bin/python" ]] || { echo "missing $VENV/bin/python (run: make venv)" >&2; exit 1; }
  render
  id -u "$SERVICE_USER" >/dev/null 2>&1 || useradd --system --user-group --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
  install -d -m 0700 -o root -g root "$SECRETS_DIR"
  install -d -m 0755 "$WEB_DIR"
  install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_USER" "$WEB_DIR/data"
  install -m 0644 "$APP_DIR/services/dashboard/index.html" "$WEB_DIR/index.html"
  local f
  for f in "$OUT"/*.service "$OUT"/*.timer; do install -m 0644 "$f" "$UNIT_DIR/"; done
  systemctl daemon-reload
  for f in "$OUT"/*.service "$OUT"/*.timer; do systemd-analyze verify "$UNIT_DIR/$(basename "$f")" || true; done
  echo "Installed units into $UNIT_DIR. Secrets expected in $SECRETS_DIR (see docs/DEPLOYMENT.md)."
  echo "Live drop-ins were NOT installed. Kill switch: touch /var/lib/wallex-arena/LIVE_DISABLED"
}

cmd="${1:-render}"; shift || true
case "$cmd" in
  render) render ;;
  install)
    install_units
    if [[ "${1:-}" == "--enable" ]]; then
      shift
      for unit in "$@"; do systemctl enable --now "$unit"; done
    fi ;;
  *) echo "usage: $0 render | install [--enable UNIT...]" >&2; exit 2 ;;
esac
