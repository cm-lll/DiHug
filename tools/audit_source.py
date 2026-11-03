"""CPU/std-lib-only release audit: no imports of scientific/model modules."""
import ast
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ROOT / 'provenance/source_files.json').read_text())
    patch_path=ROOT/'provenance/abort_boundary_patch.json'
    changes=json.loads(patch_path.read_text())['files'] if patch_path.exists() else {}
    errors, sensitive, parsed = [], [], 0
    for row in manifest['files']:
        path = ROOT / row['path']
        expected=changes.get(row['path'],{}).get('current_sha256',row['published_sha256'])
        if not path.is_file() or digest(path.read_bytes()) != expected:
            errors.append('copied-source hash mismatch: ' + row['path'])
    for name,row in changes.items():
        if not (ROOT/name).is_file() or digest((ROOT/name).read_bytes())!=row['current_sha256']:
            errors.append('boundary-fix hash mismatch: '+name)
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts)
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.is_symlink() or path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.npz', '.npy', '.pickle', '.pkl', '.zip', '.log'}:
            errors.append('excluded artifact: ' + rel)
        data = path.read_bytes()
        if len(data) > 512 * 1024 or b'\0' in data:
            errors.append('large/binary file: ' + rel)
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            errors.append('nontext: ' + rel)
            continue
        if path.suffix == '.py':
            ast.parse(text, filename=rel)
            parsed += 1
        if path.suffix == '.json':
            json.loads(text)
        for label, pattern in (
            ('private-key', r'-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----'),
            ('github-token', r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
            ('cloud-key', r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
            ('personal-absolute-path', r'/(?:home|Users)/[^/\s\'\"]+|/data\d*/lyh'),
            ('assigned-secret', r'(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*[\'\"][A-Za-z0-9_+/=-]{12,}[\'\"]'),
        ):
            for match in re.finditer(pattern, text):
                sensitive.append(dict(file=rel, line=text.count('\n', 0, match.start()) + 1, category=label))
    support = json.loads((ROOT / 'configs/schema_support.json').read_text())
    for name, row in support.items():
        n = sum(row['type_counts'])
        capacities = [row['type_counts'][a] * (row['type_counts'][a] - 1) // 2 if a == b
                      else row['type_counts'][a] * row['type_counts'][b] for a, b in row['schema']]
        queried = sum(c for c, m in zip(capacities, row['reference_family_edges']) if m > 0)
        if sum(capacities) != n * (n - 1) // 2 or n != 192 or sum(capacities) != 18336:
            errors.append(name + ': full schema capacity')
        if queried != row['queried_slots'] or 18336 - queried != row['fixed_none_slots']:
            errors.append(name + ': active query capacity')
    # README local destinations must exist. Reference-style links are checked too.
    readme = (ROOT / 'README.md').read_text()
    targets = re.findall(r'\]\(([^)]+)\)', readme) + re.findall(r'^\[[^\]]+\]:\s*(\S+)', readme, re.M)
    for target in targets:
        if '://' not in target and not (ROOT / target.split('#')[0]).exists():
            errors.append('missing README destination: ' + target)
    result = dict(status='passed' if not errors and not sensitive else 'blocked',
                  files=len(files), parsed_python_files=parsed, source_files_hash_checked=len(manifest['files']),
                  schema_capacity_checked=True, readme_links_checked=len(targets), errors=errors,
                  sensitive_findings=sensitive,
                  limits='Pattern scan and manual review; not a proof that arbitrary secrets cannot exist. No model/asset loading, forward, backward, training or generation.')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
