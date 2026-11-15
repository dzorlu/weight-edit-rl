#!/bin/bash
# Every 30 minutes: pull an RL run's logs, push them to this repo, and publish the live run page and the
# trajectory viewer to the website. Usage: scripts/publish_live.sh <exp> [max-hours]
# RUNS (optional) lists every run the status page shows, e.g. RUNS="hero3 nolib1:hero3@1"; defaults to <exp>.
# The site is published from a separate worktree of the site repo; the run page and trajectory data are committed to its main first.
EXP=$1; MAXH=${2:-16}; REPO=$(cd "$(dirname "$0")/.." && pwd); SITE=${SITE:-$HOME/repos/dzorlu.github.io}; LIVE=$HOME/repos/wt/dzorlu.github.io-live
START=$(date +%s)
[ -d "$LIVE" ] || (cd "$SITE" && git fetch -q origin && git worktree add -q --detach "$LIVE" origin/main)
while true; do
  cd "$REPO"; mkdir -p results/rl/$EXP
  for f in $(uvx -q modal volume ls glimmer-runs /rl/$EXP 2>/dev/null | grep -o "$EXP-s[0-9]*-r[0-9]*.jsonl\|steps.jsonl\|config.json"); do
    uvx -q modal volume get --force glimmer-runs /rl/$EXP/$f results/rl/$EXP/$f >/dev/null 2>&1; done
  if ! grep -rIl -E "sk-ant-|tml-[A-Za-z0-9]{20}|@gmail|/Users/" results/rl/$EXP >/dev/null; then
    git add results/rl/$EXP && git diff --cached --quiet || { git commit -q -m "results: $EXP update $(date +%H:%M)" && git push -q origin main; }
  else echo "$(date +%H:%M) skipped push: a result file matched the secret scan"; fi
  python3 viewer/build.py >/dev/null
  cd "$LIVE" && git fetch -q origin && git reset -q --hard origin/main
  cp "$REPO"/viewer/site/data/*.json posts/weight-space-programming/traces/data/
  python3 - "$REPO/viewer/site/index.html" posts/weight-space-programming/traces/index.html <<'PY'
import re, sys
new = re.search(r'<script id="index" type="application/json">(.*?)</script>', open(sys.argv[1]).read(), re.S).group(1)
p = sys.argv[2]; s = open(p).read()
s = re.sub(r'(<script id="index" type="application/json">)(.*?)(</script>)', lambda m: m.group(1) + new + m.group(3), s, flags=re.S)
open(p, "w").write(s)
PY
  mkdir -p posts/weight-space-programming/run && (cd "$REPO" && python3 viewer/run_status.py "$LIVE/posts/weight-space-programming/run" ${RUNS:-$EXP} >/dev/null)
  # Commit the run page and trajectory data to the site's main, so a publish by anyone else includes them.
  git add posts/weight-space-programming/run posts/weight-space-programming/traces
  if ! git diff --cached --quiet; then
    git -c user.name="$(git -C "$SITE" config user.name)" -c user.email="$(git -C "$SITE" config user.email)" commit -q -m "weight-edits: run page and trajectories $(date +%H:%M)" \
      && git push -q origin HEAD:main || echo "$(date +%H:%M) push to the site's main failed"
  fi
  quarto publish gh-pages --no-prompt --no-browser >/tmp/publish_live.log 2>&1 && echo "$(date +%H:%M) published" || echo "$(date +%H:%M) publish failed: $(tail -2 /tmp/publish_live.log)"
  grep -q '"event": "done"\|"event": "stopped"' "$REPO/results/rl/$EXP/steps.jsonl" 2>/dev/null && { echo "run finished"; exit 0; }
  [ $(( ($(date +%s) - START) / 3600 )) -ge $MAXH ] && { echo "max hours reached"; exit 0; }
  sleep 1800
done
