"""Unscored protocol check using only three built-in fictional training records."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from . import strong_records as transport
from benign_llm_orca import study
from benign_record_feedback import lab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=transport.MODELS, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Scheduled compute step required')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    study.SUPPORTED_MODELS = transport.MODELS
    fuse = study.Fuse(max_calls=9, max_tokens=12000, max_seconds=240)
    backend = transport.LargeBackend(args.port, fuse, args.model)
    rows = []
    try:
        for task in study.tasks('train', 3):
            candidates = []
            for role in study.ROLES:
                result = backend.infer(role, task, 0, candidates,
                                       study.seed_for(101, 'preflight', task.case_id, role))
                row = {'case_id': task.case_id, 'role': role, **result}
                rows.append(row)
                with (root/'calls.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, allow_nan=False)+'\n')
                parsed = study.parse(result['text'])
                if not result['text'].strip():
                    raise ValueError('Empty response field; stop before scored experiment')
                if role == 'selector':
                    index, mode = study.resolve_selection(result['text'], candidates)
                    if index is None:
                        raise ValueError('Unsupported selector representation; no retry')
                    row['selection_mode'] = mode
                else:
                    if parsed is None or set(parsed) != set(task.expected) or not all(
                            isinstance(value, str) for value in parsed.values()):
                        raise ValueError('Record schema incompatible; no retry')
                    candidates.append(result['text'])
        report = {'verified': True, 'scope': 'unscored harmless protocol/GPU check',
                  'backend': backend.identity, 'calls': fuse.calls, 'tokens': fuse.tokens,
                  'rows': rows, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'checks': 'nonempty response, record schema, selector representation, GPU, final token counts',
                  'not_checked': 'No accuracy threshold, policy training or task-effect claim'}
        (root/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'verified': True, 'calls': fuse.calls, 'tokens': fuse.tokens}), flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps({'error': str(exc), 'retry': False,
            'backend': backend.identity, 'calls_attempted': fuse.calls,
            'tokens_with_final_frames': fuse.tokens}, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
