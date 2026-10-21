#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
NN_DATASET_ROOT="${NN_DATASET_ROOT:-$ROOT_DIR/../nn-dataset}"
STATE_DIR="$ROOT_DIR/_work/background"
REQUIRE_INHIBITOR=0

if [[ "${1:-}" == "__worker" ]]; then
	shift
	WORKER_ROOT="$1"
	WORKER_ROS_SETUP="$2"
	WORKER_DATASET_ROOT="$3"
	WORKER_INHIBITOR="$4"
	shift 4
	cd "$WORKER_ROOT"
	source "$WORKER_ROS_SETUP"
	source "$WORKER_ROOT/.venv/bin/activate"
	export PYTHONPATH="$WORKER_ROOT:$WORKER_DATASET_ROOT${PYTHONPATH:+:$PYTHONPATH}"
	if [[ "$WORKER_INHIBITOR" == "1" ]]; then
		exec systemd-inhibit --what=handle-lid-switch:sleep --mode=block \
			--why="NN-Bot model benchmark" \
			"$WORKER_ROOT/.venv/bin/python" -m ab.bot.dataset_pipeline "$@"
	fi
	exec "$WORKER_ROOT/.venv/bin/python" -m ab.bot.dataset_pipeline "$@"
fi

if [[ "${1:-}" == "--require-lid-inhibitor" ]]; then
	REQUIRE_INHIBITOR=1
	shift
fi

if [[ $# -eq 0 ]]; then
	printf 'Usage: %s [--require-lid-inhibitor] --model MODEL [pipeline options]\n' "$0" >&2
	printf '   or: %s [--require-lid-inhibitor] --all-models [pipeline options]\n' "$0" >&2
	exit 2
fi

if [[ ! -f "$ROS_SETUP" ]]; then
	printf 'ROS setup not found: %s\n' "$ROS_SETUP" >&2
	exit 1
fi
if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
	printf 'Python venv not found. Create .venv and install project requirements first.\n' >&2
	exit 1
fi
if [[ ! -d "$NN_DATASET_ROOT/ab/nn/nn" ]]; then
	printf 'NN-Dataset architecture checkout not found: %s\n' "$NN_DATASET_ROOT" >&2
	printf 'Set NN_DATASET_ROOT to a checkout containing ab/nn/nn.\n' >&2
	exit 1
fi

INHIBITOR=0
if command -v systemd-inhibit >/dev/null 2>&1 && \
	systemd-inhibit --what=handle-lid-switch:sleep --mode=block \
		--why="NN-Bot model benchmark" true >/dev/null 2>&1; then
	INHIBITOR=1
elif [[ $REQUIRE_INHIBITOR -eq 1 ]]; then
	printf 'Cannot acquire the systemd lid/sleep inhibitor; not starting.\n' >&2
	exit 1
else
	printf 'WARNING: lid/sleep inhibition is unavailable. The job survives terminal disconnect, but laptop suspend will pause it.\n' >&2
fi

mkdir -p "$STATE_DIR"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$STATE_DIR/$RUN_ID.log"
PID_FILE="$STATE_DIR/$RUN_ID.pid"

nohup bash "$0" __worker "$ROOT_DIR" "$ROS_SETUP" "$NN_DATASET_ROOT" "$INHIBITOR" "$@" \
	>"$LOG_FILE" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf 'Started NN-Bot background run.\nPID: %s\nPID file: %s\nLog: %s\n' \
	"$PID" "$PID_FILE" "$LOG_FILE"
