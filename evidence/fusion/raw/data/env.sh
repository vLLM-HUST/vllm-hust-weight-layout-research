#!/usr/bin/env bash
# Load the existing container runtime and the persistent research environment.
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
OPSET_RESEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$OPSET_RESEARCH_ROOT/.venv/bin:$PATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_PLUGINS=ascend
unset VLLM_HUST_OPERATOR_OPTIMIZATIONS
