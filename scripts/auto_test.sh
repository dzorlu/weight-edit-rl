#!/bin/bash
# Starts test-set scoring (rl.loop::test) for every finished step of the given runs, once each.
# Usage: scripts/auto_test.sh "<exp> [<exp> ...]" [max-hours]
EXPS=$1; MAXH=${2:-12}; REPO=$(cd "$(dirname "$0")/.." && pwd); START=$(date +%s); cd "$REPO"; mkdir -p results/rl
while true; do
  for E in $EXPS; do
    uvx -q modal volume get --force glimmer-runs /rl/$E/steps.jsonl /tmp/auto_test_$E.jsonl >/dev/null 2>&1 || continue
    for k in $(python3 -c "import json,sys; [print(json.loads(l)['step']) for l in open(sys.argv[1]) if json.loads(l).get('event')=='step' and json.loads(l).get('trained')]" /tmp/auto_test_$E.jsonl); do
      mark=results/rl/.tested_$E_$k; mark=results/rl/.tested_${E}_$k
      [ -f "$mark" ] && continue
      uvx -q modal run --detach -m rl.loop::test --exp $E --steps $k --parallel 1 >/dev/null 2>&1 && touch "$mark" && echo "$(date +%H:%M) started test scoring: $E step $k"
    done
  done
  [ $(( ($(date +%s)-START)/3600 )) -ge $MAXH ] && exit 0
  sleep 600
done
