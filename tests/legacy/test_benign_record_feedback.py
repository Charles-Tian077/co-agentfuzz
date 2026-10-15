"""Deterministic mechanism fixtures, never reported as real model experiments."""
import json
import tempfile
import unittest
from pathlib import Path

from benign_llm_orca.study import tasks, digest
from benign_record_feedback.lab import (
    FAULTS, Policy, corrupt, credits, observe, partitions, run, score,
)
from benign_record_feedback.audit import verify


class Fixture:
    identity = {'backend':'unit-fixture','real_llm':False}
    def infer(self,role,task,action,candidates,call_seed):
        good = json.dumps(task.expected)
        if role == 'generator': text = good
        elif role == 'editor':
            # Deliberately action-sensitive test double, not a measured LLM gain.
            text = good if action == 2 or (action == 1 and observe(candidates[0],task) != 'valid') else candidates[0]
        else:
            index = (max(range(2),key=lambda i:score(candidates[i],task)['dense']) if action == 2 else action)
            text = json.dumps({'pick':('A','B')[index]})
        return {'text':text,'prompt_tokens':10,'completion_tokens':5,
                'gpu_vram_bytes':0,'prompt_sha256':digest([role,task.source,action,candidates]),
                'communication_prompt_tokens':0}


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.task = tasks('train',3)[0]; self.good = json.dumps(self.task.expected)

    def test_faults_are_auditable_and_repairable(self):
        for fault in FAULTS[1:]:
            bad = corrupt(self.good,fault)
            self.assertEqual(score(bad,self.task)['exact'],0)
            self.assertLess(score(bad,self.task)['dense'],1)
            self.assertEqual(score(self.good,self.task)['dense'],1)

    def test_corruption_does_not_use_expected(self):
        self.assertEqual(json.loads(corrupt('{"x":"unknown"}','wrong_value')),{'x':'fictional_mismatch'})

    def test_duplicate_keys_rejected(self):
        self.assertEqual(score('{"name":"ava","name":"ava","city":"lima"}',self.task)['dense'],0)

    def test_states_do_not_disclose_expected_values(self):
        a = observe(self.good,self.task)
        b = observe(corrupt(self.good,'wrong_value'),self.task)
        self.assertEqual(a,b)
        self.assertEqual(observe(corrupt(self.good,'missing'),self.task),'missing')
        self.assertEqual(observe(corrupt(self.good,'syntax'),self.task),'syntax')

    def test_dense_gradient_distinguishes_partial_errors(self):
        self.assertGreater(score(corrupt(self.good,'wrong_value'),self.task)['dense'],score(corrupt(self.good,'syntax'),self.task)['dense'])

    def test_editor_credit_rewards_repair_and_penalizes_regression(self):
        good = score(self.good,self.task); bad = score(corrupt(self.good,'wrong_value'),self.task)
        tokens = dict.fromkeys(('generator','editor','selector'),0)
        repaired = credits(good,bad,good,[bad,good],good,tokens)
        regressed = credits(good,good,bad,[good,bad],bad,tokens)
        self.assertGreater(repaired['editor']['reward'],.3)
        self.assertLess(regressed['editor']['reward'],0)

    def test_selector_credit_centered_on_available_candidates(self):
        good = score(self.good,self.task); bad = score('',self.task)
        tokens = dict.fromkeys(('generator','editor','selector'),0)
        self.assertGreater(credits(good,bad,good,[bad,good],good,tokens)['selector']['reward'],0)
        self.assertLess(credits(good,bad,good,[bad,good],bad,tokens)['selector']['reward'],0)

    def test_correct_maintenance_is_not_zero(self):
        good = score(self.good,self.task)
        value = credits(good,good,good,[good,good],good,dict.fromkeys(('generator','editor','selector'),0))
        self.assertAlmostEqual(value['editor']['reward'],.3)
        self.assertAlmostEqual(value['selector']['reward'],.3)

    def test_cost_uses_actual_call_tokens(self):
        good = score(self.good,self.task)
        value = credits(good,good,good,[good,good],good,dict.fromkeys(('generator','editor','selector'),1024))
        self.assertAlmostEqual(value['editor']['cost'],.02)

    def test_nonzero_real_q_update(self):
        for role in ('generator','editor','selector'):
            actor = Policy(role); update = actor.update('x',2,.4,True)
            self.assertAlmostEqual(update['new'],.08)
            self.assertEqual(actor.choose('x','contact',101,False)[0],2)

    def test_disabled_learning_does_not_update(self):
        actor = Policy('editor'); before = digest(actor.state())
        actor.update('x',2,1,False); self.assertEqual(before,digest(actor.state()))

    def test_peer_affects_actual_next_choice_without_llm_text(self):
        actor = Policy('selector')
        actor.receive({'family':'contact','focus':2,'utility':1.})
        for seed in range(10):
            self.assertEqual(actor.choose('x','contact',seed,False)[0],2)

    def test_messages_reject_text_and_wrong_family(self):
        actor = Policy('editor')
        with self.assertRaises(ValueError): actor.receive({'source':'not allowed'})
        with self.assertRaises(ValueError): actor.receive({'family':'other','focus':1,'utility':1.})

    def test_train_eval_disjoint_and_fault_coverage(self):
        cases = partitions(True)
        self.assertEqual(set(f for p,t,f in cases if p == 'train'),set(FAULTS))
        self.assertEqual(set(f for p,t,f in cases if p == 'eval'),set(FAULTS))
        train = {t.source for p,t,f in cases if p == 'train'}
        test = {t.source for p,t,f in cases if p == 'eval'}
        self.assertFalse(train & test)

    def test_end_to_end_replay_and_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'fixture'
            summary = run(Fixture(),root)
            self.assertEqual(summary['calls'],288)
            self.assertTrue(verify(root,True)['verified'])
            full = summary['trials'][0]
            self.assertTrue(all(full['q_changes'][r] > 0 for r in ('generator','editor','selector')))
            self.assertGreater(full['repair_opportunities'],0)
            self.assertGreater(full['candidate_disagreements'],0)
            self.assertEqual(full['communication_prompt_tokens'],0)
            self.assertEqual(summary['trials'][2]['communication_bytes'],0)
            with self.assertRaises(AssertionError): verify(root,False)
            rows = (root/'episodes.jsonl').read_text().splitlines()
            row = json.loads(rows[0]); row['updates'][1]['new'] += .01
            rows[0] = json.dumps(row)
            (root/'episodes.jsonl').write_text('\n'.join(rows)+'\n')
            with self.assertRaises(AssertionError): verify(root,True)


if __name__ == '__main__': unittest.main()
