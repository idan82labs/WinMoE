#!/usr/bin/env bash
set -euo pipefail
mkdir -p state/progress/session-log
log="state/progress/session-log/$(date +%F).md"
if [ ! -f "$log" ]; then
  printf '# Session log for %s

' "$(date +%F)" > "$log"
fi
printf -- '- %s: file edit/write hook fired
' "$(date -Iseconds)" >> "$log"
cat >/dev/null || true
