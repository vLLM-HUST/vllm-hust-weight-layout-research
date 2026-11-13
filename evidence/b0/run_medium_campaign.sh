#!/usr/bin/env bash
set -eo pipefail
source /root/opset-research/env.sh
set -u
export B0_DEVICES=0,1
export PYTHONUNBUFFERED=1
cd /root/opset-research/b0-weight-layout-20261023
TASK_PYTHON=/root/opset-research/.venv/bin/python
"$TASK_PYTHON" -u monitor.py >> monitor-medium.log 2>&1 &
TASK_MONITOR_PID=$!
"$TASK_PYTHON" -u monitor_metrics.py >> monitor-metrics-medium.log 2>&1 &
finish() {
  npu-smi info > final-npu-state.txt 2>&1 || true
  printf '{"controller_exited":true,"revision":"medium-v1"}\n' > cleanup.complete.json
  kill "$TASK_MONITOR_PID" 2>/dev/null || true
}
trap finish EXIT
"$TASK_PYTHON" -u run_medium.py
"$TASK_PYTHON" -u run_medium_extras.py
