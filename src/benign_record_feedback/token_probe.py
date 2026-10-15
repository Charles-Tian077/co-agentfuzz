"""Matched harmless prompt-cost probe; not a policy efficacy benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

from benign_llm_orca.study import Fuse, ROLES, digest, resolve_selection, tasks
from .lab import RecordBackend, score

class ProbeBackend(RecordBackend):
    variant = 'compact'
    history = None

    def request(self, path, payload=None):
        if path == '/api/generate':
            payload = dict(payload)
            if self.variant == 'legacy_history':
                payload['prompt'] += '\nOther roles, verified training history only: ' + json.dumps(self.history,sort_keys=True)
            self.actual_prompt_hash = digest(payload['prompt'])
        return super().request(path,payload)

    def infer(self,*args,**kwargs):
        value = super().infer(*args,**kwargs)
        value['prompt_sha256'] = self.actual_prompt_hash
        value.pop('communication_prompt_tokens')  # Measured pair difference below, not assumed zero.
        return value


def run(backend, output, prior):
    prior = Path(prior).resolve()
    if not prior.is_dir():
        raise FileNotFoundError('Provide the verified input run directory with --prior')
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    evidence = json.loads((prior/'audit.json').read_text())
    prior_summary = json.loads((prior/'summary.json').read_text())
    if evidence.get('verified') is not True or evidence.get('real_llm') is not True:
        raise ValueError('Verified real-model input required')
    if backend.identity['model_digest'] != prior_summary['manifest']['backend']['model_digest']:
        raise ValueError('Model digest changed')
    rows = [json.loads(x) for x in (prior/'episodes.jsonl').read_text().splitlines()]
    training = [r for r in rows if r['condition']=='full' and r['phase']=='train']
    heldout = [r for r in rows if r['condition']=='full' and r['phase']=='eval'][:4]
    cases = {t.case_id:t for t in tasks('eval',6,offset=16)}
    manifest = {'scope':'matched prompt-cost probe on four fictional records, twelve role inputs',
                'backend':backend.identity, 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'input_ledger_sha256':hashlib.sha256((prior/'episodes.jsonl').read_bytes()).hexdigest(),
                'change':'append versus omit numeric training-history text only; same action/input/schema/seed',
                'limits':{'calls':40,'tokens':20000,'seconds':180},
                'limitations':'not overall policy-cost savings, accuracy superiority or an attack experiment'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        pairs = []
        for row in heldout:
            task = cases[row['case_id']]
            peers = [r for r in training if r['case_id'].split('-')[1]==task.kind]
            # Explicit family filtering; all history comes from training, never evaluation.
            for step in row['steps']:
                role = step['role']; history = {}
                for sender in ROLES:
                    if sender == role: continue
                    stage = {'generator':'generated','editor':'edited','selector':'selected'}[sender]
                    history[sender] = {'n':len(peers),'mean_quality':sum(r['scores'][stage]['exact'] for r in peers)/len(peers)}
                inputs = [] if role == 'generator' else [row['draft']] if role == 'editor' else row['candidates']
                pair = {'case_id':task.case_id,'role':role,'action':step['action'],'training_history':history}
                variants = ('compact','legacy_history') if len(pairs)%2==0 else ('legacy_history','compact')
                for variant in variants:
                    backend.variant = variant; backend.history = history
                    pair[variant] = backend.infer(role,task,step['action'],inputs,101)
                    text = pair[variant]['text']
                    if role == 'selector':
                        index, mode = resolve_selection(text,inputs)
                        text = inputs[index] if index is not None else ''
                    pair[variant]['record_exact'] = score(text,task)['exact']
                pair['prompt_token_saving'] = pair['legacy_history']['prompt_tokens']-pair['compact']['prompt_tokens']
                pairs.append(pair)
                (output/'pairs.json').write_text(json.dumps(pairs,indent=2))
        totals = {v:sum(p[v]['prompt_tokens']+p[v]['completion_tokens'] for p in pairs) for v in ('compact','legacy_history')}
        result = {'status':'complete','manifest':manifest,'pairs':len(pairs),'actual_calls':len(pairs)*2,
                  'actual_total_tokens':sum(totals.values()),'variant_total_tokens':totals,
                  'prompt_token_saving':sum(p['prompt_token_saving'] for p in pairs),
                  'exact_outcomes_changed':sum(p['compact']['record_exact']!=p['legacy_history']['record_exact'] for p in pairs),
                  'all_pairs_have_positive_prompt_saving':all(p['prompt_token_saving']>0 for p in pairs)}
        (output/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'error':str(exc),'retry':False,'partial':True}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True,type=int); parser.add_argument('--output',required=True)
    parser.add_argument('--prior',required=True,help='Verified input run directory')
    args = parser.parse_args()
    run(ProbeBackend(args.port,Fuse(max_calls=40,max_tokens=20000,max_seconds=180),'deepseek-r1:7b'),args.output,args.prior)


if __name__ == '__main__':
    main()
