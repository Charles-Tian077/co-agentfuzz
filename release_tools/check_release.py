"""Local source-only hygiene gate; never uploads files or prints secret values."""
from pathlib import Path
import argparse
import ast
import json
import re
import tomllib

SKIP = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.ruff_cache',
        'build', 'dist'}
FORBIDDEN_DIRS = {'Results', 'Datasets', 'tmp', 'Logs', 'cache', 'weights', 'private', 'raw', 'outputs'}
FORBIDDEN_EXTS = {'.docx', '.pptx', '.pdf', '.png', '.jpg', '.jpeg', '.svg', '.m',
                  '.gguf', '.safetensors', '.pt', '.pth', '.ckpt', '.bin', '.pem', '.key'}
TOKEN = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b')
PRIVATE_PATH = re.compile(r'(?:[A-Za-z]:[\\/](?:Users|research)[\\/]|/deac/|/data/dw/|wfu-lovelace)')


def check(root):
    root = Path(root).resolve()
    findings = []
    count = 0
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in SKIP or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.is_symlink():
            findings.append({'path': relative.as_posix(), 'check': 'symlink'}); continue
        if path.is_dir():
            continue
        count += 1
        label = relative.as_posix()
        if any(part in FORBIDDEN_DIRS for part in relative.parts) and label != 'Datasets/README.md':
            findings.append({'path': label, 'check': 'private/generated artifact'}); continue
        if relative.parts[0] == 'models' and label not in {'models/README.md', 'models/registry.json'}:
            findings.append({'path': label, 'check': 'unreviewed model artifact'}); continue
        if relative.parts[0] == 'datasets' and label not in {
            'datasets/README.md', 'datasets/synthetic/train.jsonl',
            'datasets/synthetic/test.jsonl', 'datasets/synthetic/manifest.json',
        }:
            findings.append({'path': label, 'check': 'unreviewed dataset'}); continue
        if path.suffix.lower() in FORBIDDEN_EXTS or (path.name.startswith('.env') and path.name != '.env.example'):
            findings.append({'path': label, 'check': 'non-source or secret file'}); continue
        if path.stat().st_size > 2_000_000:
            findings.append({'path': label, 'check': 'large file'}); continue
        try:
            text = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            findings.append({'path': label, 'check': 'non-UTF8 file'}); continue
        if TOKEN.search(text):
            findings.append({'path': label, 'check': 'credential-shaped token'})
        if label != 'release_tools/check_release.py' and PRIVATE_PATH.search(text):
            findings.append({'path': label, 'check': 'private absolute path or login alias'})
        if path.suffix == '.py':
            try:
                tree = ast.parse(text, filename=label)
            except SyntaxError as exc:
                findings.append({'path': label, 'check': 'syntax', 'line': exc.lineno}); continue
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'eval'
                   for n in ast.walk(tree)):
                findings.append({'path': label, 'check': 'unsafe eval'})
        elif path.suffix == '.json':
            try: json.loads(text)
            except ValueError: findings.append({'path': label, 'check': 'invalid JSON'})
        elif path.suffix == '.toml':
            try: tomllib.loads(text)
            except ValueError: findings.append({'path': label, 'check': 'invalid TOML'})
    return {'passed': not findings, 'source_files_checked': count, 'findings': findings,
            'scope': 'static hygiene heuristic, not proof of licensing or absence of all secrets',
            'network_access': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    result = check(parser.parse_args().root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
