import at
import numpy as np
import pickle
import nlk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

ring = at.load_lattice('/machfs/sauret/SharedCodes/tracking/injection_nlk_mb_qd3.mat')
field = np.loadtxt('/machfs/sauret/tracking/kicker_fields/turn_after_nlk_field.txt', skiprows=8)
poly_b = np.polyfit(field[:,0], field[:,1]/20.16, 30)
poly_b = np.flip(poly_b)
nlk2_8 = at.Multipole('NLK', length=1, poly_a=np.zeros(len(poly_b)), poly_b=-poly_b)

with open('/machfs/sauret/SharedCodes/tracking/beam_opt2.pickle', 'rb') as f:
    beam = pickle.load(f)
beam = np.squeeze(beam[:,:,0,0])
# plt.plot(beam[0], beam[1], marker='o', ls='')
# plt.show()
# exit()
beam_out,*_ = ring.track(beam,1)
beam_out = np.squeeze(beam_out[:,:,0,0])
print(beam_out.shape)
_ = nlk2_8.track(beam_out, in_place=True)

