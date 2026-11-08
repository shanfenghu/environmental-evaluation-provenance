"""Run the standalone source-assessment test subset without an editable install."""
import os
from pathlib import Path
import sys
import unittest
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_assessment*.py'))
    raise SystemExit(0 if result.wasSuccessful() else 1)
