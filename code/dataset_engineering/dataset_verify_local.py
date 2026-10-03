"""Run original-format validation on this computer only."""
import os
from pathlib import Path
import runpy
os.environ['DATASET_ROOT'] = 'E:/datasets'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
runpy.run_path(str(Path(__file__).with_name('dataset_verify.py')), run_name='__main__')
