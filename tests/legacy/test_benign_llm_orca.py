"""Neutral fixtures only; no real model calls or original attack imports."""
import copy
import json
import io
import tempfile
import unittest
from pathlib import Path
from benign_llm_orca.study import Agent,Fuse,Ollama,Task,digest,parse,quality,resolve_selection,rewards,run,tasks,trial
from benign_llm_orca.audit import verify

class Fixture:
    identity={'real_llm':False,'model_digest':'unit-fixture-not-evidence','backend':'unit-fixture'}
    def infer(self,role,task,action,peer,candidates,call_seed):
        text=json.dumps(task.expected) if role!='selector' else '{"pick":"B"}'
        if role=='generator' and action==0: text='{}'
        return {'text':text,'prompt_tokens':10+len(peer),'completion_tokens':5,'seconds':0,'prompt_sha256':'fixture'}

class BenignStudyTests(unittest.TestCase):
    def test_heldout_records_disjoint(self):
        a=tasks('train',18); b=tasks('eval',30)
        self.assertFalse({t.source for t in a}&{t.source for t in b})
        c=tasks('eval',60,offset=16)
        self.assertFalse({t.source for t in a+b}&{t.source for t in c})
        self.assertEqual(len({t.source for t in c}),60)
    def test_partition_and_budget_validation(self):
        for phase,count in [('custom',3),('train',21),('eval',63),('eval',10)]:
            with self.assertRaises(ValueError): tasks(phase,count)
        for phase,count,offset in [('train',18,1),('eval',60,17),('eval',3,-1)]:
            with self.assertRaises(ValueError): tasks(phase,count,offset)
    def test_strict_success(self):
        task=tasks('train',3)[0]
        self.assertEqual(quality(json.dumps(task.expected),task),1)
        for text in ['{}','```json\n{}\n```','[]']:
            self.assertEqual(quality(text,task),0)
    def test_duplicate_keys_rejected(self): self.assertIsNone(parse('{"name":"ava","name":"ava"}'))
    def test_selector_protocol_and_deepseek_candidate_echo(self):
        candidates=['{"name":"ava","city":"lima"}','{"name":"mia","city":"rome"}']
        self.assertEqual(resolve_selection('{"pick":"B"}',candidates),(1,'protocol'))
        self.assertEqual(resolve_selection('{"city":"lima","name":"ava"}',candidates),(0,'candidate_echo'))
        self.assertEqual(resolve_selection(candidates[0],[candidates[0],candidates[0]]),(0,'candidate_echo'))
        self.assertEqual(resolve_selection('{"pick":"C"}',candidates),(None,'invalid'))
        self.assertEqual(resolve_selection('{"pick":"A","extra":1}',candidates),(None,'invalid'))
    def test_deepseek_selector_uses_strict_schema_and_short_output(self):
        client=Ollama.__new__(Ollama); client.model_name='deepseek-r1:7b'; client.fuse=Fuse()
        captured={}
        def request(path,payload=None):
            if path=='/api/generate':
                captured.update(payload)
                return {'done':True,'response':'{"pick":"A"}','prompt_eval_count':20,'eval_count':5}
            return {'models':[{'name':'deepseek-r1:7b','size_vram':1}]}
        client.request=request
        task=tasks('train',3)[0]
        result=client.infer('selector',task,0,{},['{}','{}'],41)
        self.assertEqual(result['text'],'{"pick":"A"}')
        self.assertEqual(captured['format']['properties']['pick']['enum'],['A','B'])
        self.assertFalse(captured['think'])
        self.assertEqual(captured['options']['num_predict'],16)
    def test_reward_credit(self): self.assertEqual(rewards(0.,1.,1.),{'generator':0.,'editor':1.,'selector':1.})
    def test_streaming_transport_preserves_final_counts(self):
        client=Ollama.__new__(Ollama); client.url='http://127.0.0.1:20001'; client.fuse=Fuse()
        frames=[{'response':'{','done':False},
                {'response':'"pick":"A"}','done':False},
                {'response':'','done':True,'prompt_eval_count':20,'eval_count':5,'done_reason':'stop'}]
        class LocalReply:
            def open(self,request,timeout):
                self.request=request
                return io.BytesIO(b''.join(json.dumps(f).encode()+b'\n' for f in frames))
        client.opener=LocalReply()
        result=client.request('/api/generate',{'stream':True})
        self.assertEqual(result['response'],'{"pick":"A"}')
        self.assertEqual((result['prompt_eval_count'],result['eval_count']),(20,5))
    def test_streaming_transport_rejects_missing_final_counts(self):
        client=Ollama.__new__(Ollama); client.url='http://127.0.0.1:20001'; client.fuse=Fuse()
        class LocalReply:
            calls=0
            def open(self,request,timeout):
                self.calls+=1
                return io.BytesIO(b'{"response":"partial","done":false}\n')
        client.opener=LocalReply()
        with self.assertRaisesRegex(ValueError,'without final token counts'):
            client.request('/api/generate',{'stream':True})
        self.assertEqual(client.opener.calls,1)
        self.assertFalse(client.failed_inference['final_frame'])
    def test_real_inbox_and_numerical_update(self):
        a=Agent('generator'); frame={'id':'x','sender':'validator','receiver':'generator','reward':1.}
        self.assertTrue(a.receive(frame)['completed']); self.assertIn('x',a.inbox)
        update=a.update('x','contact',1,True)
        self.assertEqual(update['new_q'],.2); self.assertEqual(a.version,1); self.assertFalse(a.inbox)
    def test_disabled_learning_does_not_create_q_state(self):
        a=Agent('generator'); a.receive({'id':'x','sender':'validator','receiver':'generator','reward':1.})
        a.update('x','contact',1,False); self.assertEqual(a.policy_state(),{'q':{},'version':0,'peer':{}})
    def test_duplicate_delivery_rejected(self):
        a=Agent('generator'); f={'id':'x','sender':'validator','receiver':'generator','reward':1.}
        a.receive(f)
        with self.assertRaises(ValueError): a.receive(f)
    def test_wrong_receiver_rejected(self):
        with self.assertRaises(ValueError): Agent('generator').receive({'id':'x','receiver':'selector'})
    def test_all_four_conditions_freeze_evaluation(self):
        for condition in ('full','no_learning','no_communication','neither'):
            trace=[]; result=trial(Fixture(),condition,41,tasks('train',3),tasks('eval',3),trace.append)
            self.assertTrue(result['evaluation_frozen'])
            self.assertEqual(result['update_calls'],9 if condition in ('full','no_communication') else 0)
            self.assertTrue(all(not u['enabled'] for e in trace if e['phase']=='eval' for u in e['updates']))
    def test_no_communication_has_no_peer_receipts(self):
        trace=[]; trial(Fixture(),'no_communication',41,tasks('train',3),tasks('eval',3),trace.append)
        self.assertTrue(all(not s['peer_used'] for e in trace for s in e['steps']))
        self.assertTrue(all(len(e['receipts'])==3 for e in trace))
    def test_full_communication_has_actual_next_decision_history(self):
        trace=[]; trial(Fixture(),'full',41,tasks('train',6),tasks('eval',3),trace.append)
        self.assertTrue(any(s['peer_used'] for e in trace for s in e['steps']))
        self.assertEqual(len(trace[0]['receipts']),9)
    def test_calls_fuse_without_retry(self):
        f=Fuse(max_calls=1); f.reserve()
        with self.assertRaises(RuntimeError): f.reserve()
        self.assertEqual(f.calls,1)
    def test_token_cost_missing_is_error(self):
        for value in (None,-1,True):
            with self.assertRaises(ValueError): Fuse().account(value,1)
    def test_token_limit_stops(self):
        with self.assertRaises(RuntimeError): Fuse(max_tokens=1).account(1,1)
    def test_custom_endpoint_not_supported(self):
        with self.assertRaises(ValueError): Ollama(443,Fuse())
    def test_arbitrary_model_not_supported(self):
        with self.assertRaises(ValueError): Ollama(20001,Fuse(),'arbitrary:latest')
    def test_full_ledger_replay_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'run'; run(Fixture(),folder,True)
            with self.assertRaises(AssertionError): verify(folder)
            audit=verify(folder,unit_fixture=True); self.assertEqual(audit['model_calls'],72)
            path=folder/'episodes.jsonl'; rows=[json.loads(l) for l in path.read_text().splitlines()]
            rows[0]['updates'][0]['new_q']+=1
            path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
            with self.assertRaises(AssertionError): verify(folder,unit_fixture=True)
    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError): run(Fixture(),tmp,True)

if __name__=='__main__': unittest.main()
