import at
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import copy
import runners
matplotlib.rcParams.update({'font.size': 15})

def inj_eff(beam):
    lost = np.sum(np.isnan(beam[0, :, 0, -1]))
    eff = (np.shape(beam)[1] - lost) / np.shape(beam)[1]
    return eff * 100, lost

lattice_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/newtl2.mat', check=False)

twiin = {'beta': [6.73,5.21],
         'alpha': [-2.21,1.4775],
         'ClosedOrbit': np.zeros(4),
         'dispersion': [-0.136, -0.092, 0, 0],
         'mu': [0, 0]
         }

idx_s3 = lattice_tl2.get_uint32_index('S3')[0]
_, _, l_opt = lattice_tl2.get_optics(refpts=idx_s3+1, twiss_in=twiin)
emit_x, emit_y = 30e-9,60e-9
e_spread, blength = 1.2e-3, 20e-3

end_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/end_tl2.mat', check=False)
rev_end_tl2 = end_tl2.reverse(copy=True)

orbit_rev = rev_end_tl2.find_orbit(refpts=at.End, orbit=np.array([-8.0e-3,0,0,0,0,0]))
orbit_rev[1][0][1] *= -1

nparts=1001
opt_beam = at.beam(nparts, at.sigma_matrix(twiss_in=l_opt[0], emity=emit_y, emitx=emit_x, espread=e_spread, blength=blength))
sextu = at.ThinMultipole('SEXTU', poly_b=[0,0,120], poly_a=[0,0,0])
sextu.MaxOrder = 2
_ = sextu.track(opt_beam, in_place=True)
for i in range(6):
   opt_beam[i] += orbit_rev[1][0][i]
_ = end_tl2.track(opt_beam, nturns=1, in_place=True)
plt.scatter(opt_beam[0], opt_beam[1])
plt.show()

# idx_nlk2 = end_tl2.get_uint32_index('NLK2')[0]
# idx_dr2 = end_tl2.get_uint32_index('DR_NLK2')[0]
# idx_nlk3 = end_tl2.get_uint32_index('NLK3')[0]
# s_marker = 0.5*end_tl2[idx_nlk2].Length + end_tl2[idx_dr2].Length + end_tl2[idx_nlk3].Length
ring = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/injection_nlk_mb_qd3.mat', mat_key='ring')
ring.enable_6d()
# new_ring = ring.sbreak(break_s=s_marker)
# idx_mk = new_ring.get_uint32_index('sbreak')[0]
# rot_ring = new_ring.rotate(idx_mk)
# idx_id04 = rot_ring.get_uint32_index('ID04')[0]
# o6,*_ = rot_ring.find_orbit()
o6, *_ = ring.find_orbit()


for j in range(6):
    opt_beam[j] += o6[j]

# rot_beam, *_ = rot_ring.track(opt_beam, nturns=1, refpts=idx_id04, use_mp=True)
# rot_beam = np.squeeze(rot_beam[:,:,0,0])
beam_out, *_ = ring.track(opt_beam, nturns=1001, use_mp=True)
print(beam_out.shape)

ie, lost = inj_eff(beam_out)
print(ie)