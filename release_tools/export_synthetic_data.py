"""Export existing built-in fictional fixtures; no inference, network or new tasks."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from coagent_repro import replicate_records as harness
from benign_llm_orca import study
from benign_record_feedback import lab


def payloads():
    partitions = harness.partitions(False)
    payload = {}
    for phase, filename in (('train', 'train.jsonl'), ('eval', 'test.jsonl')):
        rows = [dict(task.__dict__, phase=phase, intervention=fault)
                for split, task, fault in partitions if split == phase]
        payload[filename] = ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                             allow_nan=False) + '\n' for row in rows).encode('utf-8')
    manifest = {
        'dataset_id': 'co-agentfuzz-fictional-repair-v1',
        'scope': 'harmless fixed-record mechanism study; not an attack benchmark',
        'schema_version': 1,
        'training_records': 18, 'test_records': 60,
        'evaluation_offset': 16, 'seeds': list(harness.SEEDS),
        'families': ['contact', 'inventory', 'meeting'],
        'interventions': list(lab.FAULTS),
        'conditions': list(study.CONDITIONS),
        'partition_digest': study.digest([(p, t.__dict__, f) for p, t, f in partitions]),
        'labels_are_verifier_only': True,
        'runtime_reads_built_in_tasks': True,
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(harness.__file__), Path(study.__file__), Path(lab.__file__))},
        'files': {name: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                  for name, raw in payload.items()},
        'family_counts': {phase: dict(Counter(t.kind for p, t, _ in partitions if p == phase))
                          for phase in ('train', 'eval')},
        'provenance': 'Deterministic export of existing source fixtures, not model observations.',
    }
    payload['manifest.json'] = (json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                          indent=2, allow_nan=False) + '\n').encode('utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets' / 'synthetic')
    parser.add_argument('--check', action='store_true', help='Verify byte-identical export without writing')
    args = parser.parse_args()
    expected = payloads()
    if args.check:
        failures = [name for name, raw in expected.items()
                    if not (args.output / name).is_file() or (args.output / name).read_bytes() != raw]
        print(json.dumps({'verified': not failures, 'mismatched_files': failures, 'inference': False}))
        raise SystemExit(bool(failures))
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in expected):
        raise FileExistsError('Refusing to overwrite an existing dataset export')
    for name, raw in expected.items():
        with (args.output / name).open('xb') as handle:
            handle.write(raw)
    print(json.dumps({'written_files': list(expected), 'training_records': 18,
                      'test_records': 60, 'inference': False}))


if __name__ == '__main__':
    main()
