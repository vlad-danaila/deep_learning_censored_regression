import torch as t

LOSS = 0
ABS_ERR = 1
R_SQUARED = 2

IS_CUDA_AVILABLE = t.cuda.is_available()
GRID_RESULTS_FILE = 'grid_results.tar'

DOT_SIZE = 1