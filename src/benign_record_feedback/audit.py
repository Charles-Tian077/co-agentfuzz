"""Replay all benign decisions, labels, credits, acknowledgements and costs."""
import argparse
import hashlib
import json
from pathlib import Path

from benign_llm_orca.study import CONDITIONS, digest
from .lab import partitions, trial


def verify(folder, fixture=False):
    root = Path(folder)
    for name in ('manifest.json','episodes.jsonl','summary.json'):
        if (root/name).stat().st_size > 20000000: raise ValueError('Evidence size limit')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(x) for x in (root/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert summary['manifest'] == manifest and summary['status'] == 'complete'
    assert manifest['seed'] == 101
    assert manifest['conditions'] == {k:list(v) for k,v in CONDITIONS.items()}
    assert manifest['backend']['real_llm'] is (not fixture)
    if fixture: assert manifest['backend']['backend'] == 'unit-fixture'
    else:
        assert manifest['backend']['backend'] == 'ollama_loopback'
        assert manifest['backend']['model'] in ('deepseek-r1:7b','qwen2.5:7b')
        assert len(manifest['backend']['model_digest']) == 64
    for name, value in manifest['source_hashes'].items():
        assert Path(name).name == name
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == value
    dependency = Path(__file__).parents[1].joinpath('benign_llm_orca/study.py')
    assert hashlib.sha256(dependency.read_bytes()).hexdigest() == manifest['transport_source_sha256']
    assert manifest['partition_digest'] == digest([(p,t.__dict__,f) for p,t,f in partitions(manifest['smoke'])])
    assert len(rows) == 4*len(partitions(manifest['smoke']))
    assert len({r['episode'] for r in rows}) == len(rows)
    assert (manifest['alpha'],manifest['epsilon'],manifest['peer_weight'],manifest['cost_weight']) == (.2,.2,.15,.02)
    class Replay:
        def __init__(self, group):
            self.outputs = iter([s for row in group for s in row['steps']])
            self.used = 0
        def infer(self, role, task, action, candidates, call_seed):
            step = next(self.outputs); self.used += 1
            assert step['role'] == role and step['action'] == action
            if not fixture: assert type(step['gpu_vram_bytes']) is int and step['gpu_vram_bytes'] > 0
            return {k:step[k] for k in ('text','prompt_tokens','completion_tokens',
                    'gpu_vram_bytes','prompt_sha256','communication_prompt_tokens')}
    replayed = []
    for condition in CONDITIONS:
        group = [r for r in rows if r['condition'] == condition]
        backend = Replay(group); rebuilt = []
        value = trial(backend,condition,101,manifest['smoke'],rebuilt.append)
        assert rebuilt == group, 'Decision, reward, receipt or next-state tampering'
        assert value == next(t for t in summary['trials'] if t['condition'] == condition)
        assert backend.used == value['calls']
        replayed.append(value)
    assert sum(t['calls'] for t in replayed) == summary['calls']
    assert sum(t['tokens'] for t in replayed) == summary['tokens']
    result = {'verified':True,'episodes':len(rows),'actual_calls':summary['calls'],
              'actual_tokens':summary['tokens'],'real_llm':not fixture,
              'frozen_eval':True,'reward_and_next_decision_replayed':True}
    (root/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder'); parser.add_argument('--unit-fixture',action='store_true')
    args = parser.parse_args(); print(json.dumps(verify(args.folder,args.unit_fixture)))
