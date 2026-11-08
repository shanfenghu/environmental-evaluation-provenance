"""Archive repacking must not change which verified scientific members are read."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from source_assessment.prepare import prepare_air, prepare_hydrology, verify_raw


class PreparationTests(unittest.TestCase):
    def test_repacked_air_archive_ignores_wrapper_order_and_unlisted_csv(self):
        header='State Code,County Code,Site Num,POC,Latitude,Longitude,Observation Count,Date Local,Arithmetic Mean\n'
        value=(header+'01,001,0001,1,40,-80,24,2021-01-01,3\n'+
               '01,001,0001,1,40,-80,24,2022-01-01,4\n').encode()
        cfg={'archives':['input.zip'],'latitude_bounds':[25,50],'longitude_bounds':[-125,-66],
             'minimum_observations_per_day':18,'years':[2021,2022],
             'minimum_days_each_period':1,'geo_cut_lat':39,'geo_cut_lon':-100}
        members={'input.zip':['values.csv']}
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            with zipfile.ZipFile(root/'input.zip','w')as z:z.writestr('values.csv',value)
            expected,meta=prepare_air(root,cfg,members)
            manifest={'inputs':[{'file':'input.zip','sha256':'original-container',
                'members':[{'member':'values.csv','bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}]}]}
            with zipfile.ZipFile(root/'input.zip','w')as z:
                z.writestr('README.txt','new wrapper');z.writestr('unlisted.csv','unverified content')
                z.writestr('values.csv',value)
            self.assertFalse(verify_raw(root,manifest,True)[0]['exact_container'])
            actual,actual_meta=prepare_air(root,cfg,members)
            self.assertEqual(actual_meta,meta)
            self.assertEqual(set(actual),set(expected))
            for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
            np.testing.assert_array_equal(actual['y0'],[3.]);np.testing.assert_array_equal(actual['w0'],[4.])

    def test_hydrology_ignores_unlisted_timeseries(self):
        topo='data/CAMELS_GB_topographic_attributes.csv'
        series='data/timeseries/CAMELS_GB_hydromet_timeseries_1_19701001-20150930.csv'
        cfg={'train_start':'2000-01-30','train_end':'2000-01-30',
             'test_start':'2000-01-31','test_end':'2000-01-31',
             'minimum_coverage_each_period':1,'geo_cut_lat':53.5,'geo_cut_lon':-2.5}
        import pandas as pd
        rows='date,precipitation,temperature,pet,discharge_spec\n'+''.join(
            f'{d.date()},1,10,1,2\n'for d in pd.date_range('2000-01-01',periods=31))
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            with zipfile.ZipFile(root/'camels_gb_2020.zip','w')as z:
                z.writestr(topo,'gauge_id,gauge_lat,gauge_lon,area,elev_mean\n1,52,-3,100,50\n')
                z.writestr(series,rows)
                z.writestr(series.replace('_1_','_999_'),'unverified extra series')
            arrays,meta=prepare_hydrology(root,cfg,{'camels_gb_2020.zip':[topo,series]})
            self.assertEqual([m['id']for m in meta],['1'])
            np.testing.assert_array_equal(arrays['y0'],[2.]);np.testing.assert_array_equal(arrays['w0'],[2.])

if __name__=='__main__':unittest.main()
