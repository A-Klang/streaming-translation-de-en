#!/bin/bash
# Overnight runs for one arm: start its llama-server, run every setting on the
# report sentences, stop the server. A setting that crashes is resumed in the same
# file (the server is restarted first if it died), up to MAX_RESUMES times.
#
# Usage, from the repository folder:
#   pipeline/night.sh hymt2-q4
#   pipeline/night.sh gemma-e2b-q4
#   pipeline/night.sh hymt2-q4 grammar     (the Grammar delay sweep instead of the main settings)
# Optional: START and N override the sentence range, e.g. START=300 N=5 pipeline/night.sh hymt2-q4

set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

ARM="${1:?give an arm: hymt2-q4 or gemma-e2b-q4}"
GROUP="${2:-main}" # main: wait-k and local agreement; grammar: the Grammar delays
START="${START:-300}" # devtest 300 onwards is kept for the report runs
N="${N:-712}" # devtest 300 to 1011
MAX_RESUMES=3

case "$ARM" in
  hymt2-q4)     SERVER=(llama-server -hf tencent/Hy-MT2-1.8B-GGUF:Q4_K_M --port 8080 -c 4096 -np 1 -ngl 99) ;;
  gemma-e2b-q4) SERVER=(llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --port 8080 -c 4096 -np 1 -ngl 99 --swa-full --no-mmproj) ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac

# the settings, as simulate.py arguments, and the result folder each one writes to
SETTINGS=(
  "--policy waitk --k 1|waitk1"
  "--policy waitk --k 3|waitk3"
  "--policy waitk --k 5|waitk5"
  "--policy la --stride 1|la-s1"
  "--policy la --stride 1 --verb-rule|la-s1-verb"
  "--policy la --stride 2|la-s2"
  "--policy la --stride 2 --verb-rule|la-s2-verb"
  "--policy la --stride 3|la-s3"
  "--policy la --stride 3 --verb-rule|la-s3-verb"
)
if [ "$GROUP" = grammar ]; then
  SETTINGS=(
    "--policy grammar --delay 0|grammar-d0"
    "--policy grammar --delay 1|grammar-d1"
    "--policy grammar --delay 2|grammar-d2"
  )
elif [ "$GROUP" != main ]; then
  echo "unknown group $GROUP (main or grammar)"; exit 1
fi

LOG_DIR="logs/night_${ARM}_${GROUP}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
SERVER_PID=""

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG_DIR/night.log"; }

healthy() { curl -sf localhost:8080/health > /dev/null; }

start_server() {
  if healthy; then
    log "a server is already running on port 8080; stop it first"
    exit 1
  fi
  "${SERVER[@]}" >> "$LOG_DIR/llama-server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 120); do # wait up to 10 minutes (a first download takes a while)
    if healthy; then log "server up ($ARM, pid $SERVER_PID)"; return 0; fi
    sleep 5
  done
  log "server did not come up, see $LOG_DIR/llama-server.log"
  exit 1
}

stop_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2> /dev/null
    wait "$SERVER_PID" 2> /dev/null
    SERVER_PID=""
  fi
}
trap stop_server EXIT

start_server
log "settings: ${#SETTINGS[@]}, sentences $START to $((START + N - 1))"

for entry in "${SETTINGS[@]}"; do
  args="${entry%%|*}"
  name="${entry##*|}"
  run_log="$LOG_DIR/$name.log"
  if ! healthy; then
    log "server is down, restarting"
    stop_server
    start_server
  fi
  log "start $name"
  # shellcheck disable=SC2086
  python pipeline/simulate.py --arm "$ARM" $args --start "$START" --n "$N" >> "$run_log" 2>&1
  status=$?
  # the file this run writes to, from its own log
  out=$(grep -m1 '^writing ' "$run_log" | cut -d' ' -f2-)

  resumes=0
  while [ $status -ne 0 ] && [ -n "$out" ] && [ $resumes -lt $MAX_RESUMES ]; do
    resumes=$((resumes + 1))
    log "$name failed (exit $status), resume $resumes of $MAX_RESUMES"
    if ! healthy; then
      log "server is down, restarting"
      stop_server
      start_server
    fi
    # shellcheck disable=SC2086
    python pipeline/simulate.py --arm "$ARM" $args --start "$START" --n "$N" --resume "$out" >> "$run_log" 2>&1
    status=$?
  done

  if [ $status -eq 0 ]; then
    log "done  $name: $(wc -l < "$out" | tr -d ' ') sentences in $out"
  else
    log "GAVE UP on $name (exit $status), see $run_log"
  fi
done

log "all settings finished"
