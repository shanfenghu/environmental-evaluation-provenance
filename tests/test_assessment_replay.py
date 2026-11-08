"""Replay evidence must cover the declared cases, cycles and physical loss states."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from source_assessment.io import sha
from source_assessment.verify import compare


class ReplayTests(unittest.TestCase):
    def fixture(self, root, stage='full'):
        def write(directory, cycles, keys, actual=False):
            directory.mkdir()
            case={'full_loss_key':'full','cycles':[{'cycle':i,'folds':[{'loss_key':f'fold{i}'}]} for i in cycles], 'value':1.}
            (directory/'case.json').write_text(json.dumps(case))
            with (directory/'losses.zip').open('wb')as stream:
                np.savez_compressed(stream,**{k:np.ones((2,3))for k in keys})
            run={'complete':True,'configuration_sha256':'fixed',
                 'cases':[{'record':'case.json','sha256':sha(directory/'case.json')}],
                 'loss_archive_keys':sorted(keys),'loss_archive_sha256':sha(directory/'losses.zip')}
            if actual:run.update(stage=stage,cycle_indices=cycles)
            (directory/'run.json').write_text(json.dumps(run))
        write(root/'reference',[0,1],['full','fold0','fold1'])
        ids=[]if stage=='full_only'else[0]if stage=='fixed_cycle'else[0,1]
        write(root/'replay',ids,['full',*[f'fold{i}'for i in ids]],True)

    def mutate(self, path, fn):
        value=json.loads(path.read_text());fn(value);path.write_text(json.dumps(value))

    def test_full_and_reduced_replays_report_actual_coverage(self):
        for stage,states,full in [('full',3,True),('full_only',1,False),('fixed_cycle',2,False)]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory()as directory:
                root=Path(directory);self.fixture(root,stage)
                result=compare(root/'reference',root/'replay',root/'out.json')
                self.assertEqual(result['native_source_states_replayed'],states)
                self.assertEqual(result['full_native_run_replayed'],full)

    def test_empty_or_missing_archive_members_cannot_claim_full_replay(self):
        for keys in [[],['full','fold0']]:
            with self.subTest(keys=keys), tempfile.TemporaryDirectory()as directory:
                root=Path(directory);self.fixture(root)
                with(root/'replay/losses.zip').open('wb')as stream:np.savez_compressed(stream,**{k:np.ones((2,3))for k in keys})
                # Even a refreshed archive digest cannot replace missing physical states.
                self.mutate(root/'replay/run.json',lambda x:x.update(loss_archive_sha256=sha(root/'replay/losses.zip')))
                with self.assertRaises(ValueError):compare(root/'reference',root/'replay',root/'out.json')

    def test_scientific_mode_allows_only_version_digests_and_declared_tolerance(self):
        for problem in ['none','scientific_value','membership','tolerance']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory()as directory:
                root=Path(directory);self.fixture(root,'full_only')
                for name in ['reference','replay']:
                    case=root/name/'case.json'
                    self.mutate(case,lambda x:x.update(input_sha256=name,
                        full_query_signatures={'F':name},development=[0,1]))
                    if name=='replay':
                        self.mutate(case,lambda x:x.update(value=1.+(1e-5 if problem=='scientific_value'else 1e-9),
                            development=[0,2]if problem=='membership'else[0,1]))
                    self.mutate(root/name/'run.json',lambda x:x['cases'][0].update(sha256=sha(case)))
                with self.assertRaises(AssertionError):
                    compare(root/'reference',root/'replay',root/'exact.json')
                if problem=='none':
                    result=compare(root/'reference',root/'replay',root/'out.json',atol=1e-8,scientific=True)
                    self.assertFalse(result['input_versions'][0]['same_bytes'])
                    self.assertEqual(result['comparison_mode'],'scientific')
                else:
                    with self.assertRaises((AssertionError,ValueError)):
                        compare(root/'reference',root/'replay',root/'out.json',
                                atol=float('nan')if problem=='tolerance'else 1e-8,scientific=True)

    def test_missing_cases_cycles_or_nonfinite_values_are_rejected(self):
        for problem in ['cases','cycles','nonfinite']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory()as directory:
                root=Path(directory);self.fixture(root)
                if problem=='cases':self.mutate(root/'replay/run.json',lambda x:x.update(cases=[]))
                else:
                    case=root/'replay/case.json'
                    self.mutate(case,lambda x:x.update(cycles=[]if problem=='cycles'else x['cycles'],value=float('nan')if problem=='nonfinite'else 1.))
                    self.mutate(root/'replay/run.json',lambda x:x['cases'][0].update(sha256=sha(case)))
                with self.assertRaises(ValueError):compare(root/'reference',root/'replay',root/'out.json')

if __name__=='__main__':unittest.main()
