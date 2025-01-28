import at
import nlk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 15})

with open('/machfs/sauret/SharedCodes/tracking/end_tl2.pickle', 'rb') as f1:
    end_tl2 = pickle.load(f1)

# for e in end_tl2:
#     print(e)
# exit()

rev_endtl2 = end_tl2.reverse(copy=True)

idx_nlk = rev_endtl2.get_uint32_index('NLK*')
_, o1 = rev_endtl2.find_orbit(refpts=idx_nlk, orbit=np.array([-8.0e-3,0,0,0,0,0]))
_, o2 = rev_endtl2.find_orbit(refpts=idx_nlk+1, orbit=np.array([-8.0e-3,0,0,0,0,0]))
print(o1.shape)
print(o1[:, 1]-o2[:, 1])

