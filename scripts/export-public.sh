#!/usr/bin/env bash
# Produce a clean, single-commit copy of this repository for a public hand-over repo.
#
#   scripts/export-public.sh OUT_DIR [--rewrite FROM=TO]... [--denylist FILE] [--strict]
#
#   OUT_DIR            new directory (must not exist); becomes a fresh git repo with one commit
#   --rewrite FROM=TO  literal replacement in every text file, e.g. --rewrite moazeniz/meowmeow-=acme/
#   --denylist FILE    one literal per line that must NOT appear (server IPs, team id, account names...).
#                      Keep this file OUTSIDE git. Lines starting with # are ignored.
#   --strict           also fail on the generic warnings (public IPv4 addresses, 24-hex object ids)
#
# What it does
#   1. `git archive HEAD`: only committed files (never .env, data/, databases, private notes: .gitignore'd).
#   2. Removes paths listed in .publicignore (gitignore-style globs, optional).
#   3. Applies --rewrite pairs.
#   4. Scans the result: gitleaks (binary or docker) + denylist (hard fail) + generic patterns.
#   5. git init + one commit "Public release from <sha>"; prints the push commands. Nothing is pushed.
set -euo pipefail
die() { echo "export-public: $*" >&2; exit 1; }
[ $# -ge 1 ] || { sed -n '2,20p' "$0"; exit 2; }
OUT=$1; shift
REWRITES=() DENY="" STRICT=0
while [ $# -gt 0 ]; do
  case $1 in
    --rewrite) REWRITES+=("$2"); shift 2 ;;
    --denylist) DENY=$2; shift 2 ;;
    --strict) STRICT=1; shift ;;
    *) die "unknown option $1" ;;
  esac
done
ROOT=$(git rev-parse --show-toplevel)
SHA=$(git -C "$ROOT" rev-parse --short HEAD)
[ -e "$OUT" ] && die "$OUT already exists"
[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ] || echo "warning: uncommitted changes are NOT exported (HEAD only)"

mkdir -p "$OUT"
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$OUT"

if [ -f "$OUT/.publicignore" ]; then
  while IFS= read -r pat; do
    case "$pat" in ''|'#'*) continue ;; esac
    (cd "$OUT" && shopt -s globstar nullglob && for p in $pat; do rm -rf -- "$p" && echo "removed $p"; done)
  done < "$OUT/.publicignore"
  rm -f "$OUT/.publicignore"
fi

for pair in "${REWRITES[@]}"; do
  from=${pair%%=*} to=${pair#*=}
  grep -rlIF -- "$from" "$OUT" | while IFS= read -r f; do
    python3 - "$f" "$from" "$to" <<'PY'
import sys
p, a, b = sys.argv[1:]
s = open(p, encoding="utf-8", errors="surrogateescape").read()
open(p, "w", encoding="utf-8", errors="surrogateescape").write(s.replace(a, b))
PY
  done
  echo "rewrote '$from' -> '$to'"
done

fail=0
echo "== secret scan (gitleaks)"
if command -v gitleaks >/dev/null; then
  gitleaks dir "$OUT" --redact --no-banner || fail=1
elif command -v docker >/dev/null; then
  docker run --rm -v "$OUT:/scan:ro" zricethezav/gitleaks:v8.30.1 dir /scan --redact --no-banner || fail=1
else
  echo "warning: gitleaks not available (install it or docker); relying on pattern checks below"
fi

echo "== built-in patterns"
if grep -rInE --exclude-dir=.git -- '-----BEGIN [A-Z ]*PRIVATE KEY-----|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.|(ghp|gho|ghs)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}' "$OUT"; then
  echo "ERROR: credential-shaped strings found"; fail=1
fi
if [ -n "$DENY" ]; then
  echo "== denylist ($DENY)"
  while IFS= read -r lit; do
    case "$lit" in ''|'#'*) continue ;; esac
    if grep -rlIF --exclude-dir=.git -- "$lit" "$OUT" >/tmp/export-hits.$$; then
      echo "ERROR: denylisted value (line '${lit:0:4}...') found in:"; sed 's/^/   /' /tmp/export-hits.$$; fail=1
    fi
  done < "$DENY"
  rm -f /tmp/export-hits.$$
fi
echo "== generic review hints"
ips=$(grep -rIohE --exclude-dir=.git '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' "$OUT" | grep -vE '^(127\.|10\.|0\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|255\.|1\.2\.3\.|203\.0\.113\.|198\.51\.100\.|192\.0\.2\.|169\.254\.|100\.64\.|8\.8\.)' | sort -u || true)
[ -n "$ips" ] && { echo "public IPv4 addresses present (review): $(echo $ips | tr '\n' ' ')"; [ $STRICT = 1 ] && fail=1; }
oids=$(grep -rIohE --exclude-dir=.git '\b[0-9a-f]{24}\b' "$OUT" | sort -u | head -5 || true)
[ -n "$oids" ] && { echo "24-hex ids present (Liara object ids?): $(echo $oids | tr '\n' ' ')"; [ $STRICT = 1 ] && fail=1; }

[ $fail = 0 ] || { echo "export blocked: fix the findings above (nothing was committed)"; exit 1; }

git -C "$OUT" init -q -b main
git -C "$OUT" add -A
git -C "$OUT" -c user.name="${EXPORT_AUTHOR_NAME:-MeowMeow Team}" -c user.email="${EXPORT_AUTHOR_EMAIL:-team@example.com}" \
    commit -q -m "Public release from ${SHA}"
echo "== exported $(git -C "$OUT" ls-files | wc -l) files to $OUT (1 commit)."
echo "next: cd $OUT && git remote add origin git@github.com:<org>/<repo>.git && git push -u origin main"
