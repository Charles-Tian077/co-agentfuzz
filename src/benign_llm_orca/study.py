"""Three-role contextual-bandit validation with a real loopback-only local LLM.

The only tasks are built-in contact, inventory and meeting record extraction.
No arbitrary prompts, external datasets, plugins, operators or target endpoints.
LLM weights are frozen; learned objects are three small action-value tables.
"""
import argparse
import hashlib
import json
import math
import random
import time
import urllib.request
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROLES=('generator','editor','selector')
CONDITIONS={'full':(True,True),'no_learning':(False,True),
            'no_communication':(True,False),'neither':(False,False)}
STYLES=('Be concise and direct.','Check the required structure before answering.',
        'Check fidelity to the source before answering.')
ALPHA=.2
EPSILON=.2
SUPPORTED_MODELS=('qwen2.5:7b','deepseek-r1:7b','gpt-oss:20b')
SELECTOR_SCHEMA={
    'type':'object',
    'properties':{'pick':{'type':'string','enum':['A','B']}},
    'required':['pick'],
    'additionalProperties':False,
}

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class Task:
    case_id:str
    kind:str
    source:str
    expected:dict

def tasks(phase,count,offset=None):
    limits={'train':18,'eval':60}
    if phase not in limits or not 0<count<=limits[phase] or count%3:
        raise ValueError('Fixed task partition/count required')
    fixtures={
      'contact': [('ava','lima'),('noah','oslo'),('mia','rome'),('liam','perth'),('zoe','kyoto'),('omar','accra'),
                  ('iris','dover'),('theo','basel'),('nora','turin'),('finn','riga'),('lena','sofia'),('milo','quito'),
                  ('sara','leeds'),('hugo','porto'),('esme','nantes'),('ivan','split'),('yara','tunis'),('felix','berlin'),
                  ('hana','seoul'),('kai','helsinki'),('luca','madrid'),('maya','vienna'),('nico','prague'),('pia','warsaw'),
                  ('ravi','delhi'),('uma','surat'),('vera','ghent'),('wade','austin'),('xena','athens'),('yuri','odessa'),
                  ('zara','cairo'),('arlo','boston'),('cleo','dublin'),('dara','lagos'),('elio','milan'),('gaia','lyon')],
      'inventory':[('chair','blue'),('desk','green'),('lamp','white'),('shelf','black'),('mug','red'),('clock','yellow'),
                   ('plate','silver'),('bowl','orange'),('vase','purple'),('tray','brown'),('book','indigo'),('pen','violet'),
                   ('box','beige'),('frame','gold'),('basket','teal'),('cable','gray'),('sofa','cyan'),('table','maroon'),
                   ('mirror','coral'),('cabinet','navy'),('bottle','pink'),('curtain','cream'),('notebook','olive'),
                   ('keyboard','copper'),('helmet','amber'),('speaker','lime'),('folder','crimson'),('ribbon','aqua'),
                   ('candle','ivory'),('spoon','bronze'),('blanket','mint'),('pillow','peach'),('ladder','khaki'),
                   ('camera','magenta'),('wallet','charcoal'),('brush','scarlet')],
      'meeting':[('monday','atrium'),('tuesday','studio'),('wednesday','library'),('thursday','lounge'),('friday','garden'),('saturday','gallery'),
                 ('sunday','terrace'),('monday','annex'),('tuesday','courtyard'),('wednesday','workshop'),('thursday','cafeteria'),('friday','theater'),
                 ('saturday','pavilion'),('sunday','observatory'),('monday','archive'),('tuesday','laboratory'),('wednesday','depot'),
                 ('thursday','rotunda'),('friday','lobby'),('saturday','terminal'),('sunday','hall'),('monday','veranda'),
                 ('tuesday','plaza'),('wednesday','gymnasium'),('thursday','chapel'),('friday','museum'),('saturday','clinic'),
                 ('sunday','tower'),('monday','greenhouse'),('tuesday','arcade'),('wednesday','auditorium'),('thursday','harbor'),
                 ('friday','embassy'),('saturday','academy'),('sunday','orchard'),('monday','courthouse')]}
    if offset is None:
        offset=0 if phase=='train' else 6
    if type(offset) is not int or offset<0 or (phase=='train' and offset!=0):
        raise ValueError('Fixed task offset required')
    if any(offset+count//3>len(items) for items in fixtures.values()):
        raise ValueError('Fixed task range exceeded')
    result=[]
    for i in range(count//3):
        index=offset+i
        for kind,items in fixtures.items():
            a,b=items[index]
            keys={'contact':('name','city'),'inventory':('item','color'),'meeting':('day','location')}[kind]
            source=f'{keys[0]}: {a}. {keys[1]}: {b}.'
            result.append(Task(f'{phase}-{kind}-{index:02}',kind,source,dict(zip(keys,(a,b)))))
    return result

def parse(text):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    try:
        obj=json.loads(text,object_pairs_hook=unique)
        return obj if isinstance(obj,dict) else None
    except (ValueError,TypeError): return None

def quality(text,task):
    obj=parse(text)
    if obj is None or set(obj)!=set(task.expected): return 0.
    # Strict exact record: lowercase values, exact keys, no additional fields.
    return float(all(isinstance(obj[k],str) and obj[k]==v for k,v in task.expected.items()))

def resolve_selection(text,candidates):
    """Resolve a selector response without guessing or changing task quality.

    The normal protocol is {"pick":"A|B"}. Some reasoning models instead
    repeat the chosen candidate object. That model-specific representation is
    accepted only when it is semantically identical to a supplied candidate;
    identical candidates deterministically resolve to A.
    """
    obj=parse(text)
    if obj and set(obj)=={'pick'} and obj['pick'] in ('A','B'):
        return ('A','B').index(obj['pick']),'protocol'
    candidate_objects=[parse(candidate) for candidate in candidates]
    matches=[i for i,candidate in enumerate(candidate_objects)
             if obj is not None and candidate is not None and obj==candidate]
    if matches:
        return matches[0],'candidate_echo'
    return None,'invalid'

def rewards(generated,edited,selected):
    if any(v not in (0.,1.) for v in (generated,edited,selected)):
        raise ValueError('Independent binary labels required')
    return {'generator':generated,'editor':edited-generated,'selector':selected-generated}

class Fuse:
    def __init__(self,max_calls=1300,max_tokens=400000,max_seconds=1800):
        if not 1<=max_calls<=1300 or not 1<=max_tokens<=400000 or not 1<=max_seconds<=1800:
            raise ValueError('Hard budget exceeded')
        self.max_calls=max_calls; self.max_tokens=max_tokens; self.max_seconds=max_seconds
        self.start=time.monotonic(); self.calls=0; self.tokens=0
    def reserve(self):
        if time.monotonic()-self.start>=self.max_seconds or self.calls>=self.max_calls or self.tokens>=self.max_tokens:
            raise RuntimeError('Circuit breaker: time/call/token limit')
        self.calls+=1
    def account(self,prompt,completion):
        if any(type(v) is not int or v<0 for v in (prompt,completion)):
            raise ValueError('Missing or invalid actual model token counts')
        self.tokens+=prompt+completion
        if self.tokens>self.max_tokens: raise RuntimeError('Circuit breaker: token overshoot')

class Ollama:
    def __init__(self,port,fuse,model_name='qwen2.5:7b'):
        if type(port) is not int or not 20000<=port<50000: raise ValueError('Job-local port required')
        if model_name not in SUPPORTED_MODELS: raise ValueError('Unsupported local model')
        self.url=f'http://127.0.0.1:{port}'; self.fuse=fuse
        self.model_name=model_name
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        tags=self.request('/api/tags')
        model=next((m for m in tags['models'] if m.get('name')==model_name),None)
        if not model or not model.get('digest'): raise ValueError('Existing local model/digest required; no download')
        self.identity={'backend':'ollama_loopback','model':model_name,'model_digest':model['digest'],
                       'real_llm':True,'require_gpu':True,'temperature':0,'max_output_tokens':96,'context_limit':2048,
                       'cold_start_timeout_seconds':120,'regular_timeout_seconds':35}
    def request(self,path,payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(self.url+path,data=data,headers={'Content-Type':'application/json'})
        # First generation includes loading the model into GPU memory; this is
        # a bounded cold start, not a retry or an unrecorded inference.
        wait=120 if path=='/api/generate' and self.fuse.calls==1 else 35
        with self.opener.open(req,timeout=wait) as r:
            if path=='/api/generate':
                started=time.monotonic(); chunks=[]; byte_count=0
                for index,line in enumerate(r):
                    byte_count+=len(line)
                    if byte_count>1000000 or index>=10000 or time.monotonic()-started>wait:
                        raise ValueError('Streaming inference budget exceeded')
                    frame=json.loads(line)
                    if frame.get('error'): raise ValueError('Local inference error: '+str(frame['error'])[:300])
                    text=frame.get('response','')
                    if not isinstance(text,str): raise ValueError('Invalid streaming response text')
                    chunks.append(text)
                    if frame.get('done') is True:
                        return {**frame,'response':''.join(chunks)}
                self.failed_inference={'response_chunks':chunks,'final_frame':False}
                raise ValueError('Streaming inference ended without final token counts')
            body=r.read(1000001)
        if len(body)>1000000: raise ValueError('Oversized model response')
        return json.loads(body)
    def infer(self,role,task,action,peer,candidates,call_seed):
        common=f'Source: {task.source}\nRequired keys: {json.dumps(list(task.expected))}. Values must be lowercase source strings. No extra keys.'
        if role=='generator':
            instruction='Extract the record. Return only a JSON object.'
        elif role=='editor':
            instruction='Review the draft against the source and required keys. Return a corrected JSON record; preserve a correct draft.\nDraft: '+candidates[0]
        else:
            instruction=('Choose the record that best meets the source and required keys. '
                         'Return exactly one control object: {"pick":"A"} or {"pick":"B"}. '
                         'Do not repeat either record.\nA: '+candidates[0]+'\nB: '+candidates[1])
        # Only frozen training-history statistics, never held-out labels.
        prompt=common+'\n'+STYLES[action]+'\n'+instruction
        if peer: prompt+='\nOther roles, verified training history only: '+json.dumps(peer,sort_keys=True)
        self.fuse.reserve(); started=time.monotonic()
        system='You process harmless fictional records. Follow the record schema accurately.'
        if role=='selector':
            system+=' As selector, output only the required pick control object, never a record.'
        payload={'model':self.model_name,'prompt':prompt,'system':system,
            'stream':True,'keep_alive':'15m','options':{'temperature':0,'num_predict':16 if role=='selector' else 96,
                                                       'num_ctx':2048,'seed':call_seed,'num_thread':2}}
        if self.model_name=='deepseek-r1:7b':
            payload['format']=SELECTOR_SCHEMA if role=='selector' else 'json'; payload['think']=False
        obj=self.request('/api/generate',payload)
        if obj.get('done') is not True or not isinstance(obj.get('response'),str):
            self.failed_inference={'role':role,'case_id':task.case_id,'call':self.fuse.calls,
                                   'request':payload,'response':obj}
            raise ValueError('Incomplete local inference: '+json.dumps(obj,sort_keys=True)[:400])
        p=obj.get('prompt_eval_count'); c=obj.get('eval_count'); self.fuse.account(p,c)
        resident=next((m for m in self.request('/api/ps')['models'] if m.get('name')==self.model_name),None)
        vram=resident.get('size_vram',0) if resident else 0
        if type(vram) is not int or vram<=0:
            raise RuntimeError('Circuit breaker: model fell back to CPU; allocated GPU residency required')
        return {'text':obj['response'],'prompt_tokens':p,'completion_tokens':c,
                'seconds':time.monotonic()-started,'prompt_sha256':digest(prompt),'gpu_vram_bytes':vram}

class Agent:
    def __init__(self,role):
        self.role=role; self.q={}; self.version=0; self.inbox={}; self.peer={}
    def policy_state(self): return {'q':self.q,'version':self.version,'peer':self.peer}
    def choose(self,context,seed):
        values=self.q.get(context,[0.,0.,0.]); rng=random.Random(seed)
        exploration=rng.random()<EPSILON
        choices=list(range(3)) if exploration else [i for i,v in enumerate(values) if v==max(values)]
        return rng.choice(choices)
    def receive(self,frame):
        if frame['receiver']!=self.role or frame['id'] in self.inbox: raise ValueError('Invalid/duplicate delivery')
        self.inbox[frame['id']]=frame
        return {'frame_id':frame['id'],'receiver':self.role,'completed':True}
    def update(self,frame_id,context,action,learning):
        frame=self.inbox.pop(frame_id); reward=frame['reward']
        if frame['sender']!='validator' or not math.isfinite(reward): raise ValueError('Untrusted reward')
        old=self.q.get(context,[0.,0.,0.])[action]; before=self.version
        new=old+ALPHA*(reward-old) if learning else old
        if learning:
            self.q.setdefault(context,[0.,0.,0.])[action]=new; self.version+=1
        return {'role':self.role,'context':context,'action':action,'reward':reward,'old_q':old,'new_q':new,
                'version_before':before,'version_after':self.version,'enabled':learning}
    def receive_peer(self,frame,context):
        receipt=self.receive(frame); self.inbox.pop(frame['id'])
        key=context+'|'+frame['sender']; state=self.peer.setdefault(key,{'count':0,'sum_quality':0.})
        state['count']+=1; state['sum_quality']+=frame['quality']
        return receipt
    def peer_view(self,context):
        return {k.split('|')[1]:{'n':v['count'],'mean_quality':v['sum_quality']/v['count']}
                for k,v in self.peer.items() if k.startswith(context+'|')}

def seed_for(seed,phase,case_id,role):
    return int(digest([seed,phase,case_id,role])[:8],16)%2147483647

def trial(backend,condition,seed,train_cases,eval_cases,write,run_id='unit-test'):
    learning,communication=CONDITIONS[condition]; agents={r:Agent(r) for r in ROLES}
    counts={'successes':0,'evaluations':0,'prompt_tokens':0,'completion_tokens':0,'train_tokens':0,
            'update_calls':0,'q_changes':0,'self_receipts':0,'peer_receipts':0,
            'selector_protocol':0,'selector_candidate_recovery':0,'selector_invalid':0}
    evaluate_before=None
    for phase,cases in [('train',train_cases),('eval',eval_cases)]:
        if phase=='eval': evaluate_before=digest({r:a.policy_state() for r,a in agents.items()})
        for number,task in enumerate(cases):
            episode=f'{run_id}/{condition}/{seed}/{task.case_id}'; steps=[]; outputs=[]
            for role in ROLES:
                agent=agents[role]; action=agent.choose(task.kind,seed_for(seed,phase,task.case_id,role))
                peer=agent.peer_view(task.kind) if communication else {}
                result=backend.infer(role,task,action,peer,outputs[:2],seed_for(seed,phase,task.case_id,'model-'+role))
                if any(type(result.get(k)) is not int or result[k]<0 for k in ('prompt_tokens','completion_tokens')):
                    raise ValueError('Unknown token cost cannot be zero-filled')
                counts['prompt_tokens']+=result['prompt_tokens']; counts['completion_tokens']+=result['completion_tokens']
                if phase=='train': counts['train_tokens']+=result['prompt_tokens']+result['completion_tokens']
                steps.append({'role':role,'action':action,'context':task.kind,'policy_version_used':agent.version,
                              'policy_digest_used':digest(agent.policy_state()),'peer_used':peer,**result})
                outputs.append(result['text'])
            index,selection_mode=resolve_selection(outputs[2],outputs[:2])
            counts['selector_'+{'protocol':'protocol','candidate_echo':'candidate_recovery','invalid':'invalid'}[selection_mode]]+=1
            selected=outputs[index] if index is not None else ''
            labels=[quality(outputs[0],task),quality(outputs[1],task),quality(selected,task)]
            credit=rewards(*labels); receipts=[]; frames=[]; updates=[]
            # All outputs precede trusted labels. Synchronous real inbox delivery.
            for role in ROLES:
                frame={'id':episode+'/self/'+role,'episode':episode,'sender':'validator','receiver':role,'reward':credit[role]}
                frames.append(frame); receipts.append(agents[role].receive(frame))
            if len(receipts)!=3: raise RuntimeError('Feedback barrier incomplete')
            for step in steps:
                role=step['role']; updates.append(agents[role].update(episode+'/self/'+role,task.kind,step['action'],learning and phase=='train'))
            if phase=='train' and communication:
                for sender,score in zip(ROLES,labels):
                    for receiver in ROLES:
                        if sender==receiver: continue
                        frame={'id':episode+'/peer/'+sender+'/'+receiver,'episode':episode,'sender':sender,'receiver':receiver,'quality':score}
                        frames.append(frame); receipts.append(agents[receiver].receive_peer(frame,task.kind))
                        counts['peer_receipts']+=1
            if any(a.inbox for a in agents.values()): raise RuntimeError('Undrained feedback inbox')
            counts['self_receipts']+=3
            counts['update_calls']+=sum(u['enabled'] for u in updates)
            counts['q_changes']+=sum(u['enabled'] and abs(u['new_q']-u['old_q'])>1e-12 for u in updates)
            if phase=='eval': counts['successes']+=int(labels[2]); counts['evaluations']+=1
            write({'episode':episode,'condition':condition,'seed':seed,'phase':phase,'case_id':task.case_id,
                   'kind':task.kind,'steps':steps,'selected_index':index,'selection_mode':selection_mode,
                   'labels':labels,'frames':frames,'receipts':receipts,
                   'updates':updates,'barrier_closed':True,'policy_after':{r:digest(a.policy_state()) for r,a in agents.items()}})
    evaluate_after=digest({r:a.policy_state() for r,a in agents.items()})
    if evaluate_before!=evaluate_after: raise RuntimeError('Evaluation learning/history leakage')
    return {'condition':condition,'seed':seed,**counts,'total_tokens':counts['prompt_tokens']+counts['completion_tokens'],
            'eval_policy_before':evaluate_before,'eval_policy_after':evaluate_after,'evaluation_frozen':True,
            'final_policy':{r:a.policy_state() for r,a in agents.items()}}

def run(backend,output,smoke=False,replication_seed=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    if smoke and replication_seed is not None: raise ValueError('Smoke and replication are mutually exclusive')
    if smoke:
        train_cases=tasks('train',3); eval_cases=tasks('eval',3); seeds=[41]; evaluation_offset=6; phase='mechanism_smoke'
    elif replication_seed is not None:
        if replication_seed not in (101,137,173): raise ValueError('Unsupported replication seed')
        train_cases=tasks('train',18); eval_cases=tasks('eval',60,offset=16)
        seeds=[replication_seed]; evaluation_offset=16; phase='replication'
    else:
        train_cases=tasks('train',18); eval_cases=tasks('eval',30); seeds=[41,73]; evaluation_offset=6; phase='pilot'
    manifest={'schema_version':1,'run_id':str(uuid.uuid4()),'scope':'Fixed harmless record extraction only; not attack/RL-weight training evidence',
     'backend':backend.identity,'phase':phase,
     'conditions':CONDITIONS,'seeds':seeds,'alpha':ALPHA,'epsilon':EPSILON,
     'learning_rule':'one-step contextual bandit: Q <- Q + alpha*(verified reward - Q)',
     'reward_rule':'generator quality; editor quality minus generator; selection quality minus generator',
     'selector_adapter':{'schema':'exact pick enum A/B','candidate_echo_recovery':'semantic equality; first on tie',
                         'invalid_response':'no selection; no guessing'},
     'task_partition_digest':digest({'train':[t.__dict__ for t in train_cases],'eval':[t.__dict__ for t in eval_cases]}),
     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'training_cases':len(train_cases),'evaluation_cases':len(eval_cases),'evaluation_offset':evaluation_offset,
     'limitations':'Same three task families, held-out records; untrained no-learning control; communication is verified training-history statistics; no significance or guaranteed gain'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    try:
        trials=[]
        with (output/'episodes.jsonl').open('x',encoding='utf-8') as handle:
            def write(row):
                handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); handle.flush()
            for i,seed in enumerate(seeds):
                conditions=list(CONDITIONS)
                if i%2: conditions.reverse()
                for condition in conditions:
                    row=trial(backend,condition,seed,train_cases,eval_cases,write,manifest['run_id']); trials.append(row)
                    print(json.dumps({'phase':manifest['phase'],'backend':backend.identity['backend'],
                                      'condition':condition,'seed':seed,'correct':row['successes'],'cases':row['evaluations'],
                                      'tokens':row['total_tokens'],'updates':row['update_calls'],'q_changes':row['q_changes']},sort_keys=True),flush=True)
        summary={'manifest':manifest,'trials':trials,'model_calls':3*len(seeds)*4*(len(train_cases)+len(eval_cases)),
                 'recorded_total_tokens':sum(t['total_tokens'] for t in trials),'status':'complete'}
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
        return summary
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'status':'stopped','error_type':type(error).__name__,
          'message':str(error)[:400],'retry':False,'result_is_partial':True,
          'failed_inference':getattr(backend,'failed_inference',None)}),encoding='utf-8')
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True,type=int); parser.add_argument('--output',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--replication-seed',type=int,choices=(101,137,173))
    parser.add_argument('--model',choices=SUPPORTED_MODELS,default='qwen2.5:7b')
    args=parser.parse_args()
    if Path(args.output).exists(): raise FileExistsError(args.output)
    backend=Ollama(args.port,Fuse(),args.model)
    run(backend,args.output,args.smoke,args.replication_seed)

if __name__=='__main__': main()
