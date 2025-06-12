import at
import nlk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 15})

# with open('/machfs/sauret/SharedCodes/tracking/lattices/end_tl2.pickle', 'rb') as f1:
#     end_tl2 = pickle.load(f1)
end_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/end_tl2_5b_8mm.mat')

rev_end_tl2 = end_tl2.reverse(copy=True)
for e in rev_end_tl2:
    print(e.FamName)

# idx_nlk = rev_endtl2.get_uint32_index('NLK*')
idx_qf1sr = rev_end_tl2.get_uint32_index('QF1_SR')[0]

_, o1 = rev_end_tl2.find_orbit(refpts=idx_qf1sr, orbit=np.array([-8.0e-3,0,0,0,0,0]))
_, o2 = rev_end_tl2.find_orbit(refpts=idx_qf1sr+1, orbit=np.array([-8.0e-3,0,0,0,0,0]))
print(o1.shape)
print(o1)
print(o2)
print((o1[:, 0]+o2[:, 0])/2)

