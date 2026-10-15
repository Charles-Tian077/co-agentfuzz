"""Unscored V3.2 schema/GPU/token check; no accuracy threshold or retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from . import v32_records as transport
from benign_llm_orca import study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=transport.MODELS, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_STEP_ID')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    transport.configure()
    fuse = study.Fuse(max_calls=9, max_tokens=12000, max_seconds=1800)
    backend = transport.LargeBackend(args.port, fuse, args.model)
    rows = []
    try:
        for task in study.tasks('train', 3):
            candidates = []
            for role in study.ROLES:
                result = backend.infer(role, task, 0, candidates,
                    study.seed_for(101, 'preflight', task.case_id, role))
                row = {'case_id':task.case_id, 'role':role, **result}
                rows.append(row)
                with (root/'calls.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, allow_nan=False)+'\n')
                if not result['text'].strip():
                    raise ValueError('Empty final response; stop before scored experiment')
                if result['gpu_vram_bytes'] < 1024**3:
                    raise ValueError('Meaningful allocated GPU residency missing')
                parsed = study.parse(result['text'])
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
        (root/'audit.json').write_text(json.dumps({'verified':True,
            'scope':'unscored harmless schema/GPU/token check; no accuracy filtering',
            'backend':backend.identity, 'calls':fuse.calls, 'tokens':fuse.tokens,
            'rows':rows, 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
        print(json.dumps({'verified':True, 'calls':fuse.calls, 'tokens':fuse.tokens}), flush=True)
    except Exception as error:
        (root/'failure.json').write_text(json.dumps({'error':str(error), 'retry':False,
            'backend':backend.identity, 'attempted_calls':fuse.calls,
            'actual_tokens_with_final_frames':fuse.tokens,
            'failed_inference':getattr(backend, 'failed_inference', None)}, indent=2))
        raise


if __name__ == '__main__':
    main()
