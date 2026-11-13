"""Offline only: parse saved traces without loading the model."""
from pathlib import Path
import torch_npu
from torch_npu.profiler.profiler import analyse
root=Path('/root/opset-research/b0-weight-layout-20261023/profiles')
for p in sorted(root.glob('*/*_ascend_pt')):
    if (p/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').exists():continue
    print('ANALYSE',p,flush=True)
    analyse(str(p))
print('OFFLINE ANALYSIS DISPATCHED',flush=True)
