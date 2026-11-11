"""Independent small examples for weighting, coverage, cost and role contrasts."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
import csv
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from analyze_source_inclusion import summarise, source_average
from analyze_assessment_rankings import configured_inclusion, score_roles, direction
from run_pairing_precision import score_blocks
from analyze_pairing_precision import measurements
from source_assessment.illustration import illustration
from source_assessment.tables import render_tables


class FollowupTests(unittest.TestCase):
    def test_equal_cells_and_observed_coverage(self):
        records=[{'values':{'0':[2.]*6,'1':[8.]*6}},
                 {'values':{'0':[4.]*6,'2':[10.]*6}}]
        means, counts=source_average(records)
        self.assertEqual(counts, {0:2,1:1,2:1})
        np.testing.assert_array_equal(means[0], [3.]*6)
        values={i:np.full(6,x) for i,x in enumerate([0.,2.,4.,10.,20.])}
        summary=summarise(values, {'a':[0,1,2], 'b':[3,4]})
        np.testing.assert_array_equal(summary['source_mean'], [8.5]*6)
        np.testing.assert_array_equal(summary['top10pct_trimmed_mean'], [5.5]*6)
        with self.assertRaises(ValueError): summarise(values, {'empty':[9]})

    def test_roles_and_source_formula_agree_with_unequal_cell_counts(self):
        blocks=[[0,1,2],[3,4,5]]; cells={'a':[0,3],'b':[1,2,4,5]}
        loss=np.array([[[0.]*3,[2.]*3,[4.]*3,[5.]*3,[7.]*3,[9.]*3],
                       [[1.]*3,[3.]*3,[5.]*3,[4.]*3,[6.]*3,[8.]*3]])
        matrix, baseline=configured_inclusion(loss,list(range(6)),blocks,cells)
        h,t=score_roles(matrix)
        # Diagonal source means are 1.5 and 5.5; opposite roles are 2.5 and 6.5.
        np.testing.assert_allclose(h[:3], [3.5]*3)
        np.testing.assert_allclose(t[:3], [4.5]*3)
        np.testing.assert_allclose(baseline[:3], [-1.]*3)
        np.testing.assert_allclose(h-t, baseline)
        with self.assertRaises(ValueError): configured_inclusion(loss,list(range(6)),[[0,1,2],[2,4,5]],cells)
        np.testing.assert_array_equal(direction(np.array([-2.,1e-14,3.])),[-1,0,1])

    def test_conditional_variance_and_charged_cost(self):
        bank,order,fraction,ratio,efficiency=measurements(
            np.array([[0.],[2.],[4.]]),np.ones((3,1)),np.full(3,2.),np.full(3,3.))
        np.testing.assert_array_equal(bank,[4.]);np.testing.assert_array_equal(order,[1.])
        np.testing.assert_allclose(fraction,[.2]);self.assertEqual(ratio,1.5)
        np.testing.assert_allclose(efficiency,[5/6])

    def test_query_plans_share_losses_without_pooling_daily_rows(self):
        class Constant:
            def __init__(self, value): self.value=value
            def predict(self, x): return np.full(len(x),self.value)
        lengths=[1,3,2,6]
        data=SimpleNamespace(hydro=False,
            z=[np.zeros((n,1)) for n in lengths],
            w=[np.full(n,y) for n,y in zip(lengths,[0.,2.,4.,6.])],
            future=[np.zeros(n,dtype=int) for n in lengths],
            meta=[{'cell':c} for c in [0,1,0,1]])
        args=(data,[Constant(0.),Constant(1.)],{(0,0):0.,(1,0):0.},
              [[0,1],[2,3]],{0:[0,2],1:[1,3]})
        all_scores,all_losses=score_blocks(*args,[0,1])
        one_score,one_losses=score_blocks(*args,[0])
        np.testing.assert_array_equal(all_scores[0],[1.,1.,1.])
        np.testing.assert_array_equal(all_scores[1],[5.,4.,5.])
        np.testing.assert_array_equal(one_score[0],all_scores[0])
        for i in one_losses: np.testing.assert_array_equal(one_losses[i],all_losses[i])

    def test_allocation_headers_follow_the_coupling_refit_cohort_path(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            item={'workload':'fixture','task':'dispersed','target':[4.,5.,6.],
                  'joint':{'mean':[[0.]*5+[v] for v in [1.,2.,3.,6.]]},
                  'field_slices':{'gap':[3,4]},'disde':{'components':[[0.]*6]*3}}
            (root/'summary.json').write_text(json.dumps({'cases':[item]}))
            render_tables(root,root/'tables')
            rows=list(csv.reader((root/'tables/allocation.csv').open()))
            self.assertEqual(rows[0],['workload','task','coupling','refit','cohort','gap'])
            self.assertEqual(rows[1][2:],['+1.0000','+2.0000','+3.0000','+6.0000'])

    def test_confirmation_table_keeps_raw_changes_and_direction(self):
        # A positive relative contrast can hide a negative raw learner change.
        item={'workload':'air_quality_2021_2022','task':'dispersed',
              'target':[4.,5.,6.], 'joint':{'mean':[[0.]*6]*4},
              'field_slices':{'gap':[3,4]},'disde':{'components':[[0.]*6]*3},
              'followup':{'last_four_measured_exclusion':[-1.25,2.5,.75,-2.,1.75,3.75]}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'summary.json').write_text(json.dumps({'cases':[item]}))
            render_tables(root,root/'tables')
            rows=list(csv.reader((root/'tables/exclusion-means.csv').open()))
            self.assertEqual(rows[0],['workload','task','panel','cycles','unit','ridge','trees','seasonal'])
            self.assertEqual(rows[1],['air_quality_2021_2022','dispersed','original_confirmation',
                                      '4','ppb','-1.2500','+2.5000','+0.7500'])

    def test_constructed_histories_have_opposite_matched_contrasts(self):
        _,states,_,ordinary,swaps,_,_=illustration()
        self.assertEqual([len(s) for s in states],[6,4,4])
        self.assertEqual(ordinary[0],ordinary[1])
        self.assertGreater(swaps[0],0);self.assertLess(swaps[1],0)

if __name__=='__main__': unittest.main()
