#!/bin/bash
# Scores every finished RL step's submitted edits, once each, on the 2 GPUs the RL runs leave free.
#   1. MATH-500 (rl.loop::test), then 2. AIME 2024 (rl.loop::evaluate --edits, scoring only).
# Priority: regenerating a new starting model's cached answers blocks the next step, so a step is scored only once the
# following step's rollouts have started (or the run has ended), and at most 2 scoring jobs run at once.
# BACKFILL lists earlier steps to score on AIME 2024 when nothing else is waiting, as "<exp>:<k>" (k counts from 0).
# Usage: BACKFILL="hero3:0 hero3:1 nolib1:0 nolib1:1" scripts/auto_test.sh "<exp> [<exp> ...]" [max-hours]
EXPS=$1; MAXH=${2:-12}; REPO=$(cd "$(dirname "$0")/.." && pwd); START=$(date +%s); cd "$REPO"; M=results/rl/.scored; mkdir -p $M
jobs() { ls $M/*.running 2>/dev/null | while read f; do a=$(cat $f); uvx -q modal app list 2>/dev/null | grep "${a:0:14}" | grep -q ephemera && echo $f || rm -f $f; done | wc -l; }
launch() {  # name, command...
  local name=$1; shift; [ -f $M/$name.done ] && return 0; [ "$(jobs)" -ge 2 ] && return 1
  out=$("$@" 2>&1); app=$(echo "$out" | grep -o 'ap-[A-Za-z0-9]*' | head -1)
  [ -n "$app" ] && { echo $app > $M/$name.running; touch $M/$name.done; echo "$(date +%H:%M) started $name ($app)"; } || echo "$(date +%H:%M) failed to start $name: $(echo "$out" | tail -1)"
}
edits() { python3 -c "
import json,sys; recs=[json.loads(l) for l in open(sys.argv[1])]; s=[r for r in recs if r.get('event')=='step' and r['step']==int(sys.argv[2])][0]
print(','.join('rl/'+sys.argv[3]+'/work/'+x['rid'] for x in s['rollouts'] if x.get('ok'))); print(s.get('base') or 'original')" $1 $2 $3; }
while true; do
  for E in $EXPS; do
    uvx -q modal volume get --force glimmer-runs /rl/$E/steps.jsonl /tmp/auto_test_$E.jsonl >/dev/null 2>&1 || continue
    ended=$(grep -c '"event": "done"\|"event": "stopped"' /tmp/auto_test_$E.jsonl)
    for k in $(python3 -c "import json,sys; [print(json.loads(l)['step']) for l in open(sys.argv[1]) if json.loads(l).get('event')=='step']" /tmp/auto_test_$E.jsonl); do
      nxt=$(uvx -q modal volume ls glimmer-runs /rl/$E 2>/dev/null | grep -c "$E-s$((k+1))-r")
      [ "$nxt" -gt 0 ] || [ "$ended" -gt 0 ] || continue
      launch "${E}_s${k}_math500" uvx -q modal run --detach -m rl.loop::test --exp $E --steps $k --parallel 1
      read -r ED BASE <<< "$(edits /tmp/auto_test_$E.jsonl $k $E | tr '\n' ' ')"
      launch "${E}_s${k}_aime2024" uvx -q modal run --detach -m rl.loop::evaluate --name ${E}-s${k}-aime2024 --edits "$ED" --base "$BASE" --dataset aime2024 --parallel 1
    done
  done
  for b in $BACKFILL; do E=${b%:*}; k=${b#*:}
    uvx -q modal volume get --force glimmer-runs /rl/$E/steps.jsonl /tmp/auto_test_$E.jsonl >/dev/null 2>&1 || continue
    read -r ED BASE <<< "$(edits /tmp/auto_test_$E.jsonl $k $E | tr '\n' ' ')"
    launch "${E}_s${k}_aime2024" uvx -q modal run --detach -m rl.loop::evaluate --name ${E}-s${k}-aime2024 --edits "$ED" --base "$BASE" --dataset aime2024 --parallel 1
  done
  [ $(( ($(date +%s)-START)/3600 )) -ge $MAXH ] && exit 0
  sleep 600
done
