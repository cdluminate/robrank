import os
import sys
import json
import numpy as np

with open(sys.argv[1], 'rt') as f:
    j = json.load(f)
a = np.array(j)
print('shape', a.shape)
print('min', a.min())
print('mean', a.mean())
print('max', a.max())
print('std', a.std())
