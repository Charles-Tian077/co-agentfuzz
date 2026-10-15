"""Bounded read-only replay of the fixed benign study's full decision ledger."""
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from .study import tasks, digest, resolve_selection, seed_for

def verify(folder,unit_fixture=False):
    folder=Path(folder)
    for name in ('manifest.json','summary.json','episodes.jsonl'):
        if (folder/name).stat().st_size>20000000: raise ValueError('Oversized evidence')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows)>500: raise ValueError('Episode hard limit')
    assert manifest==summary['manifest'] and summary['status']=='complete'
    if unit_fixture:
        assert manifest['backend']['real_llm'] is False and manifest['backend']['backend']=='unit-fixture'
    else:
        assert manifest['backend']['real_llm'] is True and manifest['backend']['backend']=='ollama_loopback'
        assert manifest['backend']['model'] in ('qwen2.5:7b','deepseek-r1:7b','gpt-oss:20b')
        assert len(manifest['backend']['model_digest'])==64
    assert manifest['source_sha256']==hashlib.sha256(Path(__file__).with_name('study.py').read_bytes()).hexdigest()
    expected_conditions={'full':(True,True),'no_learning':(False,True),'no_communication':(True,False),'neither':(False,False)}
    assert manifest['conditions']=={k:list(v) for k,v in expected_conditions.items()}
    evaluation_offset=manifest.get('evaluation_offset',6)
    cases={t.case_id:t for t in tasks('train',manifest['training_cases'])}
    cases.update({t.case_id:t for t in tasks('eval',manifest['evaluation_cases'],offset=evaluation_offset)})
    assert len({r['episode'] for r in rows})==len(rows)
    states={}; totals={}; order={}; frame_ids=set(); changed=defaultdict(int); updates=defaultdict(int)
    receipts=defaultdict(int); phase_counts=defaultdict(int); outputs_by_case=defaultdict(list)
    role_names=('generator','editor','selector')
    for row in rows:
        condition=row['condition']; seed=row['seed']; key=(condition,seed)
        learning,communication=expected_conditions[condition]
        assert seed in manifest['seeds'] and row['case_id'] in cases
        task=cases[row['case_id']]; phase=row['phase']; assert phase in ('train','eval')
        assert row['episode']==f"{manifest['run_id']}/{condition}/{seed}/{task.case_id}"
        assert row['barrier_closed'] is True and row['kind']==task.kind
        assert task.case_id.startswith(phase+'-')
        expected_order=[t.case_id for t in tasks('train',manifest['training_cases'])]
        expected_order.extend(t.case_id for t in tasks('eval',manifest['evaluation_cases'],offset=evaluation_offset))
        index=order.get(key,0); assert row['case_id']==expected_order[index]; order[key]=index+1
        actors=states.setdefault(key,{r:{'q':{},'version':0,'peer':{}} for r in role_names})
        counts=totals.setdefault(key,{'successes':0,'evaluations':0,'prompt_tokens':0,'completion_tokens':0,
                                     'train_tokens':0,'update_calls':0,'q_changes':0,'self_receipts':0,'peer_receipts':0,
                                     'selector_protocol':0,'selector_candidate_recovery':0,'selector_invalid':0})
        if phase=='eval' and phase_counts[(key,phase)]==0: counts['eval_before']=digest(actors)
        assert [s['role'] for s in row['steps']]==list(role_names)
        for step in row['steps']:
            if not unit_fixture: assert type(step['gpu_vram_bytes']) is int and step['gpu_vram_bytes']>0
            role=step['role']; actor=actors[role]; context=task.kind
            assert step['context']==context and step['policy_version_used']==actor['version']
            assert step['policy_digest_used']==digest(actor)
            values=actor['q'].get(context,[0.,0.,0.]); rng=random.Random(seed_for(seed,phase,task.case_id,role))
            explore=rng.random()<manifest['epsilon']
            options=list(range(3)) if explore else [i for i,v in enumerate(values) if v==max(values)]
            assert step['action']==rng.choice(options)
            peer={k.split('|')[1]:{'n':v['count'],'mean_quality':v['sum_quality']/v['count']}
                  for k,v in actor['peer'].items() if k.startswith(context+'|')} if communication else {}
            assert step['peer_used']==peer
            for field in ('prompt_tokens','completion_tokens'):
                assert type(step[field]) is int and step[field]>=0
                counts[field]+=step[field]
                if phase=='train': counts['train_tokens']+=step[field]
        # Independent strict JSON and label calculation, including duplicate keys.
        def strict(text):
            def pairs(items):
                obj={}
                for k,v in items:
                    if k in obj: raise ValueError('duplicate')
                    obj[k]=v
                return obj
            try:
                obj=json.loads(text,object_pairs_hook=pairs)
                return obj if isinstance(obj,dict) else None
            except (ValueError,TypeError): return None
        texts=[s['text'] for s in row['steps']]
        selected_index,selection_mode=resolve_selection(texts[2],texts[:2])
        assert selected_index==row['selected_index']
        assert selection_mode==row['selection_mode']
        counts['selector_'+{'protocol':'protocol','candidate_echo':'candidate_recovery','invalid':'invalid'}[selection_mode]]+=1
        chosen=texts[selected_index] if selected_index is not None else ''
        labels=[float(strict(text)==task.expected) for text in [texts[0],texts[1],chosen]]
        assert row['labels']==labels
        credits={'generator':labels[0],'editor':labels[1]-labels[0],'selector':labels[2]-labels[0]}
        expected_frames=[]
        for role in role_names:
            expected_frames.append({'id':row['episode']+'/self/'+role,'episode':row['episode'],
                                    'sender':'validator','receiver':role,'reward':credits[role]})
        if communication and phase=='train':
            for sender,score in zip(role_names,labels):
                for receiver in role_names:
                    if receiver!=sender:
                        expected_frames.append({'id':row['episode']+'/peer/'+sender+'/'+receiver,'episode':row['episode'],
                                                'sender':sender,'receiver':receiver,'quality':score})
        assert row['frames']==expected_frames
        assert len(row['receipts'])==len(expected_frames)
        for frame,receipt in zip(expected_frames,row['receipts']):
            assert frame['id'] not in frame_ids; frame_ids.add(frame['id'])
            assert receipt=={'frame_id':frame['id'],'receiver':frame['receiver'],'completed':True}
            kind='self' if frame['sender']=='validator' else 'peer'; receipts[kind]+=1
            counts[kind+'_receipts']+=1
        assert [u['role'] for u in row['updates']]==list(role_names)
        for u,step in zip(row['updates'],row['steps']):
            actor=actors[u['role']]; action=step['action']; enabled=learning and phase=='train'
            old=actor['q'].get(task.kind,[0.,0.,0.])[action]
            expected=old+manifest['alpha']*(credits[u['role']]-old) if enabled else old
            assert u['enabled']==enabled and u['reward']==credits[u['role']]
            assert u['context']==task.kind and u['action']==action and u['old_q']==old
            assert math.isfinite(u['new_q']) and math.isclose(u['new_q'],expected,abs_tol=1e-12)
            assert u['version_before']==actor['version'] and u['version_after']==actor['version']+int(enabled)
            if enabled:
                actor['q'].setdefault(task.kind,[0.,0.,0.])[action]=expected; actor['version']+=1
                updates[u['role']]+=1; counts['update_calls']+=1
                if abs(expected-old)>1e-12: changed[u['role']]+=1; counts['q_changes']+=1
        for frame in expected_frames[3:]:
            peer=actors[frame['receiver']]['peer']; name=task.kind+'|'+frame['sender']
            value=peer.setdefault(name,{'count':0,'sum_quality':0.})
            value['count']+=1; value['sum_quality']+=frame['quality']
        assert row['policy_after']=={r:digest(a) for r,a in actors.items()}
        if phase=='eval':
            counts['evaluations']+=1; counts['successes']+=int(labels[2])
            outputs_by_case[task.case_id].append({'condition':condition,'seed':seed,'success':int(labels[2])})
            assert digest(actors)==counts['eval_before']
        phase_counts[(key,phase)]+=1
    assert set(states)=={(c,s) for c in expected_conditions for s in manifest['seeds']}
    assert len(summary['trials'])==len(states)
    seen=set()
    for item in summary['trials']:
        key=(item['condition'],item['seed']); assert key not in seen; seen.add(key)
        actual=totals[key]
        assert all(item[k]==v for k,v in actual.items() if k!='eval_before')
        assert phase_counts[(key,'train')]==manifest['training_cases'] and phase_counts[(key,'eval')]==manifest['evaluation_cases']
        assert item['final_policy']==states[key]
        assert item['eval_policy_before']==item['eval_policy_after']==actual['eval_before']
        assert item['total_tokens']==actual['prompt_tokens']+actual['completion_tokens']
    assert summary['model_calls']==len(rows)*3
    assert summary['recorded_total_tokens']==sum(i['total_tokens'] for i in summary['trials'])
    return {'status':'passed','unit_fixture':unit_fixture,'scope':'Full fixed benign ledger; no attack or efficacy generalization',
            'episodes':len(rows),'model_calls':len(rows)*3,'unique_eval_cases':len(outputs_by_case),
            'updates_by_role':dict(updates),'nonzero_q_changes_by_role':dict(changed),'receipts':dict(receipts),
            'frozen_evaluation':True,'next_decision_state_linkage':True,'numerical_reward_update_consistency':True,
            'source_hashes':{n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ('manifest.json','summary.json','episodes.jsonl')},
            'limitations':'Synchronous in-process receipt/replay consistency, not distributed delivery proof; bandit tables, not LLM-weight training; descriptive pilot only'}

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('folder'); args=parser.parse_args()
    result=verify(args.folder); path=Path(args.folder)/'audit.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
