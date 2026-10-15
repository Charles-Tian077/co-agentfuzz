"""Fixed-record replication harness; preserves the existing feedback algorithm.

Only built-in fictional records, the existing six fault types, three fixed seeds,
and cached loopback models are accepted. This is not an attack benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from benign_llm_orca import study
from benign_record_feedback import lab

MODELS = ('qwen2.5:7b', 'deepseek-r1:7b', 'gpt-oss:20b', 'llama3.1:8b')
SEEDS = (101, 137, 173)


def partitions(smoke):
    training = study.tasks('train', 18)
    evaluation = study.tasks('eval', 6 if smoke else 60, offset=16)
    assert not {t.source for t in training} & {t.source for t in evaluation}
    return [('train', t, lab.FAULTS[i//3]) for i,t in enumerate(training)] + [
        ('eval', t, lab.FAULTS[(i//3) % 6]) for i,t in enumerate(evaluation)]


def source_hashes():
    files = [Path(__file__), Path(lab.__file__), Path(study.__file__)]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def verify(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    rows = [json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
    assert summary['status'] == 'complete' and summary['manifest'] == manifest
    assert manifest['source_hashes'] == source_hashes()
    assert manifest['partition_digest'] == study.digest([(p,t.__dict__,f) for p,t,f in partitions(manifest['smoke'])])
    assert manifest['seed'] in SEEDS and manifest['backend']['model'] in MODELS
    assert manifest['backend']['real_llm'] is True
    assert len(manifest['backend']['model_digest']) == 64
    assert len(rows) == 4*len(partitions(manifest['smoke']))
    assert len({r['episode'] for r in rows}) == len(rows)

    class Replay:
        def __init__(self, group):
            self.steps = iter(s for r in group for s in r['steps'])
            self.used = 0
        def infer(self, role, task, action, candidates, call_seed):
            step = next(self.steps)
            assert step['role'] == role and step['action'] == action
            assert type(step['gpu_vram_bytes']) is int and step['gpu_vram_bytes'] > 0
            for k in ('prompt_tokens', 'completion_tokens'):
                assert type(step[k]) is int and step[k] >= 0
            self.used += 1
            return {k:step[k] for k in ('text', 'prompt_tokens', 'completion_tokens',
                'gpu_vram_bytes', 'prompt_sha256', 'communication_prompt_tokens')}

    for condition in study.CONDITIONS:
        group = [r for r in rows if r['condition'] == condition]
        replay = Replay(group)
        rebuilt = []
        result = lab.trial(replay, condition, manifest['seed'], manifest['smoke'], rebuilt.append)
        assert rebuilt == group, 'Feedback, reward, decision or frozen-state mismatch'
        assert result == next(t for t in summary['trials'] if t['condition'] == condition)
        assert replay.used == result['calls']
    assert summary['calls'] == sum(t['calls'] for t in summary['trials'])
    assert summary['tokens'] == sum(t['tokens'] for t in summary['trials'])
    audit = {'verified':True, 'episodes':len(rows), 'actual_calls':summary['calls'],
        'actual_tokens':summary['tokens'], 'reward_replayed':True,
        'next_decision_and_peer_state_replayed':True, 'frozen_evaluation':True,
        'gpu_residency_every_call':True,
        'artifact_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                           for n in ('manifest.json','episodes.jsonl','summary.json')}}
    (root/'audit.json').write_text(json.dumps(audit,indent=2))
    return audit


def run(args):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('A scheduled compute-node step is required')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    smoke = args.mode == 'smoke'
    fuse = study.Fuse(max_calls=350 if smoke else 1000,
                      max_tokens=100000 if smoke else 250000,
                      max_seconds=360 if smoke else 900)
    # Compatibility allowlist only; transport, prompts, rewards and policies unchanged.
    study.SUPPORTED_MODELS = MODELS
    backend = lab.RecordBackend(args.port, fuse, args.model)
    manifest = {'scope':'controlled harmless repair-space replication, not attack evidence',
        'backend':backend.identity, 'smoke':smoke, 'seed':args.seed,
        'job_id':os.environ['SLURM_JOB_ID'], 'conditions':study.CONDITIONS,
        'source_hashes':source_hashes(),
        'partition_digest':study.digest([(p,t.__dict__,f) for p,t,f in partitions(smoke)]),
        'training_records':18, 'evaluation_records':6 if smoke else 60,
        'fault_assignment':'six fixed interventions; identical across conditions/seeds',
        'alpha':lab.ALPHA, 'epsilon':lab.EPSILON, 'peer_weight':lab.PEER_WEIGHT,
        'cost_weight':lab.COST_WEIGHT,
        'changes':'experimental replication seed/count/model allowlist only; existing lab.trial and rewards unchanged',
        'limits':{'calls':fuse.max_calls,'tokens':fuse.max_tokens,'seconds':fuse.max_seconds},
        'limitations':'fixed fictional intervention panel; not natural errors, attack ASR, or LLM weight training'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        with (root/'episodes.jsonl').open('x') as handle:
            def write(row):
                handle.write(json.dumps(row,allow_nan=False)+'\n')
                handle.flush()
            trials = [lab.trial(backend,c,args.seed,smoke,write) for c in study.CONDITIONS]
        summary = {'status':'complete','manifest':manifest,'trials':trials,
            'calls':sum(t['calls'] for t in trials),
            'tokens':sum(t['tokens'] for t in trials)}
        (root/'summary.json').write_text(json.dumps(summary,indent=2))
        audit = verify(root)
        print(json.dumps({'status':'complete','model':args.model,'seed':args.seed,
            'mode':args.mode,'calls':summary['calls'],'tokens':summary['tokens'],
            'audit':audit['verified']}),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps({'error':str(exc),'retry':False,
            'partial':True,'actual_calls_attempted':fuse.calls,
            'actual_tokens_with_final_frames':fuse.tokens,
            'failed_inference':getattr(backend,'failed_inference',None)},indent=2))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--mode',choices=('smoke','replication'),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    lab.partitions = partitions  # Dataset sizing only, recorded and replayed above.
    run(args)


if __name__ == '__main__':
    main()
