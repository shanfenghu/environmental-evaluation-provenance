"""Run source-aware assessment tools from this codebase without an editable install."""
import os
from pathlib import Path
import sys

# The frozen numerical recipe fixes native learner and BLAS/OpenMP parallelism.
# Set these before importing NumPy/scikit-learn, including in isolated Python mode.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from source_assessment.cli import main

if __name__=='__main__':main()
