#!/bin/bash
# Mirrors an RL run's files and prints a short status. Usage: rl_watch.sh <exp> <app-id> [max-minutes]
EXP=$1; APP=$2; MAXM=${3:-300}; OUT=results/rl/$EXP; mkdir -p $OUT; START=$(date +%s)
while true; do
  uvx -q modal volume get --force glimmer-runs /rl/$EXP/steps.jsonl $OUT/steps.jsonl >/dev/null 2>&1
  for f in $(uvx -q modal volume ls glimmer-runs /rl/$EXP 2>/dev/null | grep -o "$EXP-s[0-9]*-r[0-9]*.jsonl"); do uvx -q modal volume get --force glimmer-runs /rl/$EXP/$f $OUT/$f >/dev/null 2>&1; done
  if grep -q '"event": "done"\|"event": "stopped"' $OUT/steps.jsonl 2>/dev/null; then echo DONE; exit 0; fi
  if uvx -q modal app list 2>/dev/null | grep "${APP:0:10}" | grep -q stopped; then echo "APP STOPPED before the run finished"; uvx -q modal app logs $APP 2>&1 | grep -E "Error|Traceback|File \"/root" | tail -6; exit 1; fi
  EL=$(( ($(date +%s)-START)/60 )); if [ $EL -gt $MAXM ]; then uvx -q modal app stop -y $APP; echo "TIMEOUT: stopped the app"; exit 2; fi
  sleep 300
done
