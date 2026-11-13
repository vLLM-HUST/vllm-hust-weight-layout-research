"""Copy historical evidence without modifying sources; does not run NPU experiments."""
from pathlib import Path
import hashlib, json, shutil, sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parent
rows = []

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def copy(src, dest):
    target = ROOT / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    sha = digest(src)
    assert sha == digest(target)
    rows.append({'path': dest.as_posix(), 'source_workspace_path': src.relative_to(WORKSPACE).as_posix(),
                 'bytes': src.stat().st_size, 'sha256': sha, 'transformation': 'none; byte-identical copy'})

def tree(src, dest, accept=lambda p: True):
    for p in sorted((WORKSPACE / src).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts and accept(p):
            copy(p, Path(dest) / p.relative_to(WORKSPACE / src))

fusion = 'opset-benchmark-20261013/reports/opset-repair-20261014'
tree(fusion, 'evidence/fusion', lambda p: 'profile' not in p.parts and not p.name.endswith('.tar.gz'))
tree('opset-deployment-20261013/route-b-20261015', 'evidence/graph-switch', lambda p: not p.name.endswith('.tar.gz'))
tree('b0-weight-layout-20261023', 'evidence/b0', lambda p: not p.name.endswith('.tar.gz') and
     not ('cells' in p.parts and any(s.startswith('large-') and s != 'large-quality' for s in p.parts)))
tree('w1-weight-layout-20261026', 'evidence/w1', lambda p: not p.name.endswith('.tar.gz'))
for name in ['AGENTS.md', 'EXPERIMENT_DEFAULTS.md']:
    p = WORKSPACE / name
    if p.exists(): copy(p, Path('evidence/conventions') / name)
for name in ['manifest.json', 'README.md', 'build_dataset.py']:
    p = WORKSPACE / 'mmlu-weight-layout-v1' / name
    if p.exists(): copy(p, Path('evidence/dataset') / name)
for split in ['small', 'medium']:
    tree('mmlu-weight-layout-v1/' + split, 'evidence/dataset/' + split)

# Original complete archive includes the large profiler traces/DBs omitted from Git.
archive = WORKSPACE / fusion / 'raw/opset-repair-evidence-20261014.tar.gz'
copy(archive, Path('release-assets') / archive.name)
assert digest(archive) == '91a8967a208da284d13ce4c749332f238f3e58e3679ffe64299889c283eaab8d'
(ROOT / 'provenance.json').write_text(json.dumps({
    'note': 'SHA-256 verifies byte-identical copies; original experiment timestamps are preserved in source records.',
    'files': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'files': len(rows), 'bytes': sum(r['bytes'] for r in rows)}, indent=2))
