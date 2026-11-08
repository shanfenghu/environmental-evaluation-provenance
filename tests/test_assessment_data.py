"""Loader checks for constant-column removal, chronology and source identities."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from source_assessment.data import Data


class DataTests(unittest.TestCase):
    def test_prepared_arrays_keep_source_roles_and_drop_only_the_fixed_constant(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);arrays={};meta=[]
            for i in range(4):
                arrays.update({f'x{i}':np.array([[1.,i,2.]]),f'z{i}':np.array([[1.,i,3.]]),
                               f'y{i}':np.array([1.]),f'w{i}':np.array([2.]),
                               f't{i}':np.array([100],dtype='int32'),f's{i}':np.array([101],dtype='int32')})
                meta.append({'id':str(i),'cell':i%2})
            path=root/'air_quality_2021_2022.npz';np.savez_compressed(path,**arrays)
            path.with_suffix('.json').write_text(json.dumps({'entities':meta,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
            data=Data('air_quality_2021_2022',root,{'air_quality_2021_2022':{'external':[3]}})
            self.assertEqual(data.dev,{0,1,2});self.assertEqual(data.cells,{0:[0,2],1:[1]})
            np.testing.assert_array_equal(data.x[2],[[2.,2.]])
            np.testing.assert_array_equal(data.y[2],[1.])
            self.assertLess(data.last_training_day,data.first_scoring_day)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):Data('air_quality_2021_2022',root,{'air_quality_2021_2022':{'external':[3]}})


if __name__=='__main__':unittest.main()
