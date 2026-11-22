import itertools
import random
import unittest

from lee_z4_dcc_comparison import (code_suite, decode, lee, prepare, raw_syndrome,
                                    sample_sphere, sphere_table, received_representative,
                                    Arithmetic)


class Z4LeeComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codes = code_suite(mothers=1)[:4]

    def test_exact_conditional_weight(self):
        for e in itertools.product(range(4),repeat=5):
            s = sum(x % 2 for x in e)
            outside = sum(x == 2 for x in e)
            self.assertEqual(lee(e),s+2*outside)

    def test_both_complete_decoders_on_entire_small_balls(self):
        errors = [[0]*10]
        for j in range(10):
            for symbol in (1,2,3):
                e = [0]*10
                e[j] = symbol
                errors.append(e)
        for j,k in itertools.combinations(range(10),2):
            for a,b in itertools.product((1,3),repeat=2):
                e = [0]*10
                e[j],e[k] = a,b
                errors.append(e)
        self.assertEqual(len(errors),211)
        for code, description in self.codes:
            for method in ('dcc6','super_support'):
                prepared = prepare(code,method)
                for index,e in enumerate(errors):
                    answer,_ = decode(prepared,raw_syndrome(prepared['h'],e),2,19000+index,False)
                    self.assertEqual(answer,e,(code.name,method,index))

    def test_public_representative_satisfies_full_syndrome(self):
        rng = random.Random(3498)
        for code,_ in self.codes:
            prepared = prepare(code,'super_support')
            for _ in range(16):
                x = [rng.randrange(4) for _ in range(code.n)]
                target = raw_syndrome(prepared['h'],x)
                y = received_representative(prepared,target,Arithmetic(2))
                self.assertEqual(raw_syndrome(prepared['h'],y),target)
            for word in code.words():
                self.assertFalse(any(raw_syndrome(prepared['h'],word)))

    def test_exact_sphere_sampler(self):
        for n,t in ((3,2),(4,3)):
            expected = [v for v in itertools.product(range(4),repeat=n) if lee(v)==t]
            table = sphere_table(n,t)
            self.assertEqual(table[n][t],len(expected))
            rng = random.Random(87)
            for _ in range(100):
                self.assertEqual(lee(sample_sphere(n,t,rng,table)),t)

    def test_instrumentation_does_not_change_search(self):
        for code,_ in self.codes:
            for method in ('dcc6','super_support'):
                prepared = prepare(code,method)
                e=[1,0,3]+[0]*(code.n-3)
                target=raw_syndrome(prepared['h'],e)
                counted,_=decode(prepared,target,2,541,True)
                timed,_=decode(prepared,target,2,541,False)
                self.assertEqual(counted,timed)


if __name__ == '__main__':
    unittest.main()
