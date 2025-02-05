import at
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import copy
import runners
matplotlib.rcParams.update({'font.size': 15})

def inj_eff(beam):
    beam0 = beam[0]
    lost = np.sum(np.isnan(beam0[0, :, 0, -1]))
    eff = (np.shape(beam0)[1] - lost) / np.shape(beam0)[1]
    return eff * 100, lost

lattice_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/newtl2.mat')

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

end_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/end_tl2.mat')
rev_end_tl2 = end_tl2.reverse(copy=True)

orbit_rev = rev_end_tl2.find_orbit(refpts=at.End, orbit=np.array([-8.0e-3,0,0,0,0,0]))
orbit_rev[1][0][1] *= -1
# print(orbit_rev)

nparts=11
opt_beam = at.beam(nparts, at.sigma_matrix(twiss_in=l_opt[0], emity=emit_y, emitx=emit_x, espread=e_spread, blength=blength),
                   orbit=orbit_rev[1][0])

nturns = 1000
x = np.arange(-2e-3,5.25e-3,0.25e-3)
xp = np.arange(-1e-3,1.25e-3,0.25e-3)
xg,xpg = np.meshgrid(x,xp)
all_beams_errors = []
for i in range(len(x)):
    for j in range(len(xp)):
        bi = copy.deepcopy(opt_beam)
        bi[0] += x[i]
        bi[1] += xp[j]
        _ = end_tl2.track(bi,nturns=1,in_place=True)
        all_beams_errors.append(bi)

runner = runners.Injeff_runner_EBS(all_beams_errors, '/machfs/sauret/SharedCodes/tracking/injection_nlk_mb_qd3.mat', 
                                   '/machfs/sauret/SharedCodes/tracking/end_tl2.mat', nturns)
runner.slurm.clean()
ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(x), len(xp))).T

pickle.dump({'x':x, 'xp':xp, 'ie':ie}, open('data_EBS1_debug.pkl', 'wb'))


