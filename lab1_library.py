
import numpy as np 
### initialize
from collections import namedtuple

# n_bits = 4;
# max_q = (2 ** (n_bits -1)) -1;
# min_q = -2 ** (n_bits -1);

# print(n_bits, min_q, max_q)

def quantize(a, n_bits):
    max_q = (2 ** (n_bits -1)) -1
    min_q = -2 ** (n_bits -1)

    return np.clip(np.round(a),min_q,max_q).astype(int)

