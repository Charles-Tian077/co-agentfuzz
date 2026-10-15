"""Auditable repair-space probe for frozen LLMs and three discrete policies.

This deliberately corrupted, harmless microbenchmark is not a natural-task,
PromptFuzz, or jailbreak benchmark. No arbitrary records/prompts are accepted.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from benign_llm_orca.study import (
    CONDITIONS, Fuse, Ollama, ROLES, SELECTOR_SCHEMA, digest, parse,
    quality, resolve_selection, seed_for, tasks,
)

FAULTS = ('clean', 'missing', 'extra', 'wrong_value', 'wrong_type', 'syntax')
FOCUS = ('Keep the response short.', 'Check JSON syntax, keys and string types.',
         'Check every value against the source.')
ALPHA, EPSILON, PEER_WEIGHT, COST_WEIGHT = .2, .2, .15, .02


def observe(text, task):
    """Only observable structure, never trusted expected values, enters a policy."""
    obj = parse(text)
    if obj is None:
        return 'syntax'
    keys = set(task.expected)  # Public required schema, not the answer values.
    flags = []
    if keys - set(obj): flags.append('missing')
    if set(obj) - keys: flags.append('extra')
    if any(not isinstance(v, str) for v in obj.values()): flags.append('type')
    return '+'.join(flags) or 'valid'


def score(text, task):
    """Verifier-only dense score; exact correctness remains the primary outcome."""
    obj = parse(text)
    if obj is None:
        return {'parse': 0., 'schema': 0., 'fields': 0., 'exact': 0., 'dense': 0.}
    schema = float(set(obj) == set(task.expected) and all(isinstance(v, str) for v in obj.values()))
    fields = sum(isinstance(obj.get(k), str) and obj[k] == v for k, v in task.expected.items()) / len(task.expected)
    return {'parse': 1., 'schema': schema, 'fields': fields,
            'exact': quality(text, task), 'dense': .1 + .2 * schema + .7 * fields}


def corrupt(text, fault):
    """Perturb the model output, not an oracle answer; record every intervention."""
    if fault not in FAULTS: raise ValueError('Fixed corruption menu required')
    if fault == 'clean': return text
    if fault == 'syntax': return text[:-1] if text.endswith('}') else text + '{'
    obj = parse(text)
    if obj is None or not obj: return text
    obj = dict(obj); key = sorted(obj)[-1]
    if fault == 'missing': obj.pop(key)
    elif fault == 'extra': obj['unused'] = 'fictional'
    elif fault == 'wrong_value': obj[key] = 'fictional_mismatch'
    elif fault == 'wrong_type': obj[key] = 0
    return json.dumps(obj, separators=(',', ':'))


def credits(generated, draft, edited, candidates, selected, tokens):
    """Dense local utility plus measured marginal credit, less actual call cost.

    These are stage differences, not Shapley values or policy-invariant shaping.
    A valid preserved record earns maintenance credit; unnecessary regressions
    and poor selections are penalized. No reward for feedback delivery itself.
    """
    g, d, e, s = (x['dense'] for x in (generated, draft, edited, selected))
    baseline = sum(x['dense'] for x in candidates) / 2
    raw = {'generator': 2*g - 1,
           'editor': .7*(e-d) + .3*(2*e-1),
           'selector': .7*(s-baseline) + .3*(2*s-1)}
    return {role: {'utility': raw[role], 'cost': COST_WEIGHT*tokens[role]/1024,
                   'reward': raw[role] - COST_WEIGHT*tokens[role]/1024}
            for role in ROLES}


class Policy:
    def __init__(self, role):
        self.role = role; self.q = {}; self.peer = {}; self.version = 0

    def state(self): return {'q': self.q, 'peer': self.peer, 'version': self.version}

    def choose(self, context, family, seed, training):
        q = self.q.get(context, [0., 0., 0.]); prior = self.peer.get(family, [0., 0., 0.])
        combined = [v + PEER_WEIGHT*p for v, p in zip(q, prior)]
        def select(values):
            rng = random.Random(seed)
            if training and rng.random() < EPSILON: return rng.randrange(3)
            return rng.choice([i for i, v in enumerate(values) if v == max(values)])
        return select(combined), select(q), {'q': list(q), 'prior': list(prior)}

    def update(self, context, action, reward, enabled):
        if not math.isfinite(reward): raise ValueError('Nonfinite verifier reward')
        old = self.q.get(context, [0., 0., 0.])[action]
        new = old + ALPHA*(reward-old) if enabled else old
        if enabled:
            self.q.setdefault(context, [0., 0., 0.])[action] = new
            self.version += 1
        return {'context': context, 'action': action, 'reward': reward,
                'old': old, 'new': new, 'enabled': enabled}

    def receive(self, message):
        # Numeric sufficient statistics only: no transcript/source/answer text.
        if set(message) != {'family', 'focus', 'utility'}: raise ValueError('Bad compact message')
        if message['family'] not in ('contact', 'inventory', 'meeting'): raise ValueError('Unknown family')
        if type(message['focus']) is not int or not 0 <= message['focus'] < 3: raise ValueError('Bad focus')
        if not math.isfinite(message['utility']) or not -2 <= message['utility'] <= 2: raise ValueError('Bad utility')
        values = self.peer.setdefault(message['family'], [0., 0., 0.])
        focus = message['focus']; values[focus] += ALPHA*(message['utility']-values[focus])


class RecordBackend(Ollama):
    """Same loopback/GPU/token fail-closed transport, shorter record-only prompts."""
    def __init__(self, port, fuse, model_name):
        super().__init__(port, fuse, model_name)
        self.identity.update(max_output_tokens=64, selector_max_output_tokens=16,
                             record_protocol='exact object schema', peer_prompt_tokens=0)

    def infer(self, role, task, action, candidates, call_seed):
        if role not in ROLES or type(action) is not int or not 0 <= action < 3:
            raise ValueError('Fixed role/action required')
        common = 'Source: ' + task.source + '\nKeys: ' + ','.join(task.expected) + '. Lowercase strings, no extra keys.\n'
        if role == 'generator': instruction = 'Extract JSON.'
        elif role == 'editor': instruction = 'Correct the draft using the source; preserve it if correct.\nDraft: ' + candidates[0]
        else: instruction = 'Select the source-faithful record. Return only {"pick":"A"} or {"pick":"B"}.\nA: ' + candidates[0] + '\nB: ' + candidates[1]
        prompt = common + FOCUS[action] + '\n' + instruction
        schema = SELECTOR_SCHEMA if role == 'selector' else {
            'type': 'object', 'properties': {k: {'type': 'string'} for k in task.expected},
            'required': list(task.expected), 'additionalProperties': False,
        }
        self.fuse.reserve()
        payload = {'model': self.model_name, 'prompt': prompt,
                   'system': 'Process harmless fictional records only.', 'stream': True,
                   'format': schema, 'think': False, 'keep_alive': '15m',
                   'options': {'temperature': 0, 'num_predict': 64 if role != 'selector' else 16,
                               'num_ctx': 2048, 'num_thread': 2, 'seed': call_seed}}
        obj = self.request('/api/generate', payload)
        self.fuse.account(obj.get('prompt_eval_count'), obj.get('eval_count'))
        if obj.get('done') is not True or not isinstance(obj.get('response'), str):
            raise ValueError('Incomplete inference')
        resident = next((m for m in self.request('/api/ps')['models'] if m.get('name') == self.model_name), None)
        vram = resident.get('size_vram', 0) if resident else 0
        if type(vram) is not int or vram <= 0: raise ValueError('GPU residency missing')
        return {'text': obj['response'], 'prompt_tokens': obj['prompt_eval_count'],
                'completion_tokens': obj['eval_count'], 'gpu_vram_bytes': vram,
                'prompt_sha256': digest(prompt), 'communication_prompt_tokens': 0}


def partitions(smoke):
    # Three kinds x six interventions, balanced, before any outcome is observed.
    train = tasks('train', 18)
    evaluate = tasks('eval', 6 if smoke else 18, offset=16)
    return [('train', t, FAULTS[i//3]) for i, t in enumerate(train)] + [
        ('eval', t, FAULTS[i % 6 if smoke else i//3]) for i, t in enumerate(evaluate)]


def trial(backend, condition, seed, smoke, write):
    learning, communication = CONDITIONS[condition]
    agents = {r: Policy(r) for r in ROLES}; frozen = None
    counts = {'correct': 0, 'evaluations': 0, 'tokens': 0, 'training_tokens': 0,
              'calls': 0, 'communication_bytes': 0, 'communication_prompt_tokens': 0,
              'peer_decision_changes': 0, 'repair_opportunities': 0, 'repairs': 0,
              'candidate_disagreements': 0, 'q_changes': {r: 0 for r in ROLES}}
    for phase, task, fault in partitions(smoke):
        training = phase == 'train'
        if not training and frozen is None: frozen = digest({r:a.state() for r,a in agents.items()})
        texts = []; steps = []; draft = ''; candidates = []
        for role in ROLES:
            if role == 'generator': context = task.kind
            elif role == 'editor': context = task.kind + '|' + observe(draft, task)
            else:
                context = task.kind + '|' + '|'.join(observe(x, task) for x in candidates)
                context += '|same' if parse(candidates[0]) == parse(candidates[1]) else '|different'
            actor = agents[role]
            action, without_peer, values = actor.choose(context, task.kind, seed_for(seed, phase, task.case_id, role), training)
            result = backend.infer(role, task, action, [] if role == 'generator' else [draft] if role == 'editor' else candidates,
                                   seed_for(seed, phase, task.case_id, 'model-'+role))
            cost = result['prompt_tokens'] + result['completion_tokens']
            if any(type(result[k]) is not int or result[k] < 0 for k in ('prompt_tokens','completion_tokens')):
                raise ValueError('Actual token counts required')
            counts['tokens'] += cost; counts['calls'] += 1
            if training: counts['training_tokens'] += cost
            counts['peer_decision_changes'] += action != without_peer
            steps.append({'role': role, 'context': context, 'action': action, 'without_peer': without_peer,
                          'values': values, 'state_before': digest(actor.state()), **result})
            texts.append(result['text'])
            if role == 'generator': draft = corrupt(texts[0], fault)
            elif role == 'editor':
                candidates = [draft, texts[1]]
                # Randomize presentation independently of condition/policy; no label-based choice.
                if seed_for(seed, phase, task.case_id, 'presentation') % 2: candidates.reverse()
        index, mode = resolve_selection(texts[2], candidates)
        chosen = candidates[index] if index is not None else ''
        scores = {name: score(text, task) for name, text in (
            ('generated',texts[0]), ('draft',draft), ('edited',texts[1]), ('selected',chosen))}
        candidate_scores = [score(x, task) for x in candidates]
        credit = credits(scores['generated'], scores['draft'], scores['edited'], candidate_scores,
                         scores['selected'], {s['role']:s['prompt_tokens']+s['completion_tokens'] for s in steps})
        updates = []; receipts = []
        episode = f'{condition}/{seed}/{task.case_id}'
        for step in steps:
            role = step['role']
            receipts.append({'id':episode+'/'+role, 'receiver':role, 'kind':'self', 'ack':True})
            update = agents[role].update(step['context'], step['action'], credit[role]['reward'], learning and training)
            updates.append({'role':role, **update})
            counts['q_changes'][role] += update['enabled'] and abs(update['new']-update['old']) > 1e-12
        messages = []
        if training and communication:
            # Shared, clipped focus evidence has the same semantics across roles.
            for step in steps:
                sender = step['role']
                utility = credit[sender]['utility']
                if abs(utility) < .05: continue  # Event gate suppresses uninformative delivery.
                message = {'family':task.kind, 'focus':step['action'], 'utility':round(utility, 4)}
                for receiver in ROLES:
                    if receiver == sender: continue
                    frame = {'id':episode+'/'+sender+'/'+receiver, 'sender':sender, 'receiver':receiver, 'message':message}
                    if len(json.dumps(frame, separators=(',', ':')).encode()) > 256: raise ValueError('Message budget')
                    agents[receiver].receive(message); messages.append(frame)
                    counts['communication_bytes'] += len(json.dumps(frame, separators=(',', ':')).encode())
                    receipts.append({'id':frame['id'], 'receiver':receiver, 'kind':'peer', 'ack':True})
        opportunity = scores['draft']['exact'] == 0
        counts['repair_opportunities'] += opportunity
        counts['repairs'] += opportunity and scores['edited']['exact'] == 1
        counts['candidate_disagreements'] += candidate_scores[0]['dense'] != candidate_scores[1]['dense']
        if not training:
            counts['correct'] += int(scores['selected']['exact']); counts['evaluations'] += 1
        write({'episode':episode, 'condition':condition, 'seed':seed, 'phase':phase,
               'case_id':task.case_id, 'fault':fault, 'draft':draft, 'candidates':candidates,
               'steps':steps, 'selection_index':index, 'selection_mode':mode,
               'scores':scores, 'candidate_scores':candidate_scores, 'credits':credit,
               'updates':updates, 'messages':messages, 'receipts':receipts, 'barrier_closed':True,
               'state_after':{r:digest(a.state()) for r,a in agents.items()}})
    final = {r:a.state() for r,a in agents.items()}
    if frozen != digest(final): raise ValueError('Evaluation changed learning/history state')
    return {'condition':condition, 'seed':seed, **counts, 'evaluation_frozen':True,
            'frozen_digest':frozen, 'final_state':final}


def run(backend, output, smoke=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    seed = 101
    manifest = {'schema':1, 'scope':'controlled harmless repair-space diagnostic; not attack evidence',
                'backend':backend.identity, 'smoke':smoke, 'seed':seed,
                'alpha':ALPHA, 'epsilon':EPSILON, 'peer_weight':PEER_WEIGHT, 'cost_weight':COST_WEIGHT,
                'partition_digest':digest([(p,t.__dict__,f) for p,t,f in partitions(smoke)]),
                'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                'transport_source_sha256':hashlib.sha256(Path(__file__).parents[1].joinpath('benign_llm_orca/study.py').read_bytes()).hexdigest(),
                'conditions':CONDITIONS,
                'limitations':'small single-seed intervention study; frozen LLM weights; contextual bandit, not full MARL or Shapley reproduction'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        with (output/'episodes.jsonl').open('x', encoding='utf-8') as handle:
            def write(row): handle.write(json.dumps(row, allow_nan=False)+'\n'); handle.flush()
            trials = [trial(backend,c,seed,smoke,write) for c in CONDITIONS]
        result = {'status':'complete', 'manifest':manifest, 'trials':trials,
                  'calls':sum(t['calls'] for t in trials), 'tokens':sum(t['tokens'] for t in trials)}
        (output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        return result
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'error':str(exc), 'retry':False,
             'partial':True, 'failed_inference':getattr(backend,'failed_inference',None)}), encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=('qwen2.5:7b','deepseek-r1:7b'), required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if Path(args.output).exists(): raise FileExistsError(args.output)
    backend = RecordBackend(args.port, Fuse(max_calls=500,max_tokens=150000,max_seconds=600), args.model)
    run(backend, args.output, not args.pilot)


if __name__ == '__main__': main()
