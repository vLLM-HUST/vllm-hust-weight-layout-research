"""Parse collected worker-daemon traces offline, including MindStudio DB."""
import json
from pathlib import Path
from torch_npu.profiler.profiler import analyse

root=Path(__file__).resolve().parent
suite=json.loads(root.joinpath('matched-suite.json').read_text())
for state in suite['runs']:
    if state['round']!=3: continue
    directory=root/'results'/state['run_id']/'profile'
    if directory.exists():
        analyse(str(directory),max_process_number=2,export_type=['text','db'])
        print('PARSED '+state['run_id'],flush=True)
