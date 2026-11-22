"""Independent checks of archived MMT success-mode reconstruction."""
import itertools
import random
import unittest

from analyze_lee_mmt_search_modes import (binary_columns, effective_weights,
    independent, phase_record)
from lee_list_isd import Parameters, Work, list_search, raw_mod_syndrome
from lee_revision import eliminate
from lee_z4_dcc_comparison import Arithmetic


class SearchModeTests(unittest.TestCase):
    def test_binary_rank_matches_unit_elimination(self):
        matrix=[[1,1,0,2],[0,2,1,3],[2,0,2,1]]
        columns=binary_columns(matrix)
        for count in range(4):
            for selected in itertools.permutations(range(4),count):
                for bits in (1,2):
                    actual=eliminate(matrix,[0]*3,selected,Arithmetic(bits,False))
                    self.assertEqual(independent(columns,selected),actual is not None)

    def test_effective_weights_respect_schedule_radius_and_capacity(self):
        p=dict(weight=1,weight2=2)
        self.assertEqual(effective_weights(p,6,5,0),(1,2))
        self.assertEqual(effective_weights(p,6,5,3),(0,0))
        self.assertEqual(effective_weights(p,5,5,0),(0,0))
        self.assertEqual(effective_weights(p,6,1,0),(1,1))

    def test_reconstruction_matches_real_search_and_accepted_free_vector(self):
        matrix=[[1,0,1,0,1,1,0,1],[0,1,1,0,1,0,1,1],
                [0,0,0,1,1,1,1,0],[0,0,1,1,0,1,0,1]]
        params=Parameters(0,1,1,0)
        for modulus,phase in ((2,'first'),(4,'native')):
            for seed in range(4):
                work=Work(seconds=10,trials=1000)
                work.phase=phase
                target=raw_mod_syndrome(matrix,[1,0,0,0,0,0,0,0],modulus)
                answer=list_search(matrix,target,4,random.Random(seed),modulus,4,'mmt',params,work)
                row=dict(counters=dict(work.stats),parameters=dict(window=0,weight=1,weight2=1,split=0))
                result=phase_record(row,phase,matrix,4,4,seed,answer,modulus)
                self.assertEqual(result['trials'],work.stats[phase+'_trials'])
                self.assertIn(result['mode'],('positive_free','nonzero_base_zero_free','zero_base'))


if __name__=='__main__':
    unittest.main()
