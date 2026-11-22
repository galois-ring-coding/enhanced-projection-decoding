"""Checks for the fresh fixed-size MMT confirmation protocol."""
from dataclasses import asdict
import unittest

from lee_list_isd import Parameters
from lee_extended import composition_probability
from run_lee_list_retuned import choose, expected_records, target_id
from run_lee_mmt_stability import TARGETS, expected_training, score, shortlist
from analyze_lee_mmt_stability import infer, trial_count


class MMTStabilityTests(unittest.TestCase):
    def test_trial_count_does_not_double_count_diagnostic_subsets(self):
        counters=dict(first_trials=10, first_singular_trials=2, first_full_rank_trials=8,
                      first_zero_list_trials=3, first_nonzero_list_trials=5,
                      recovery_trials=4, recovery_full_rank_trials=4, native_trials=0)
        self.assertEqual(trial_count(counters),14)

    def test_shortlist_is_common_and_includes_both_policies_and_zero(self):
        target = (64,12,4,3,'mmt')
        ident = target_id(target)
        params = [Parameters(), Parameters(2,0,1), Parameters(4,1,1,2), Parameters(6,1,1,2)]
        tuning = {}
        for side in ('ccp','lee'):
            scores = [dict(parameters=asdict(p), failed_calls=0, selection_loss_ns=j if side=='ccp' else 4-j)
                      for j,p in enumerate(params)]
            tuning[ident+':'+side+'_mmt'] = dict(candidates=scores)
        pool = shortlist(tuning, target)
        self.assertEqual({tuple(p.values()) for p in pool}, {tuple(asdict(p).values()) for p in params})
        self.assertEqual(len(pool),len(params))

    def test_score_weights_compositions_errors_and_seeds(self):
        block = [dict(a=a, status='complete', times={'online_ns':(a+1)*1000000})
                 for m in range(2) for a in range(3) for e in range(2) for s in range(2)]
        value = score(block,64,4,2,15)
        expected = sum(float(composition_probability(64,4,a))*(a+1)*1000000 for a in range(3))
        self.assertAlmostEqual(value['selection_loss_ns'],expected)
        self.assertEqual(value['failed_calls'],0)
        block[0]['status']='budget'
        self.assertGreater(score(block,64,4,2,15)['selection_loss_ns'],value['selection_loss_ns'])

    def test_fixed_targets_and_full_test_sample(self):
        self.assertEqual(len(TARGETS),6)
        self.assertTrue(all(t[-1]=='mmt' for t in TARGETS))
        config=dict(test_mothers=32,train_mothers=8,targets=TARGETS,
                    grids={target_id(t):[{}]*3 for t in TARGETS})
        self.assertEqual(expected_records(config),12800)
        self.assertEqual(expected_training(config),8*3*2*4*26)

    def test_family_and_positive_selection_can_differ(self):
        scores=[dict(parameters=asdict(Parameters(2,0,1)),failed_calls=0,selection_loss_ns=1),
                dict(parameters=asdict(Parameters(6,1,1,2)),failed_calls=0,selection_loss_ns=2)]
        self.assertEqual(choose(scores,'mmt',False)['weight'],0)
        self.assertEqual(choose(scores,'mmt',True)['weight'],1)

    def test_global_inference_keeps_all_lengths_and_policies(self):
        points=[]
        for n, t, k0 in ((64,4,3),(96,6,5)):
            ident=target_id((n,3*n//16,t,k0,'mmt'))
            for variant,ratio in (('primary',.6),('family',1.4)):
                for mother in range(8):
                    for side in ('ccp','lee'):
                        points.append(dict(target=ident,variant=variant,mother=mother,method=side+'_mmt',
                            n=n,online_ms=(mother+10)*(ratio if side=='ccp' else 1),all_complete=True))
        rows,protocol=infer(points,repetitions=1000)
        self.assertEqual(protocol['comparisons'],4)
        self.assertEqual(len(rows),4)
        for row in rows:
            self.assertEqual(row['assessment'],'CCP faster' if row['variant']=='primary' else 'native faster')


if __name__=='__main__':
    unittest.main()
