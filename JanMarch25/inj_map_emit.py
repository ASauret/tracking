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

lattice_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/newtl2_7b_qf1sr_vf.mat', check=False)

twiin = {'beta': [6.73,5.21],
         'alpha': [-2.21,1.4775],
         'ClosedOrbit': np.zeros(4),
         'dispersion': [-0.136, -0.092, 0, 0],
         'mu': [0, 0]
         }

idx_s3 = lattice_tl2.get_uint32_index('S3')[0]
_, _, l_opt = lattice_tl2.get_optics(refpts=idx_s3+1, twiss_in=twiin)
e_spread, blength = 1.2e-3, 20e-3

end_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/end_tl2_7b_qf1sr_vf.mat', check=False)
rev_end_tl2 = end_tl2.reverse(copy=True)

orbit_rev = rev_end_tl2.find_orbit(refpts=at.End, orbit=np.array([-8.0e-3,0,0,0,0,0]))
orbit_rev[1][0][1] *= -1

nparts=1001
nturns = 2048

ex = np.linspace(1e-9,60e-9,15)
ey = np.linspace(1e-9,80e-9,15)
all_beams_emit = []
for i in range(len(ex)):
    print(i)
    for j in range(len(ey)):
        print(j)
        opt_beam = at.beam(nparts, at.sigma_matrix(twiss_in=l_opt[0], emity=ey[j], emitx=ex[i], espread=e_spread, blength=blength),
                   orbit=orbit_rev[1][0])
        _ = end_tl2.track(opt_beam,nturns=1,in_place=True, use_mp=True)
        all_beams_emit.append(opt_beam)


runner = runners.Injeff_runner_EBS(all_beams_emit, '/machfs/sauret/SharedCodes/tracking/lattices/injection_nlk_mb.mat', 
                                   '/machfs/sauret/SharedCodes/tracking/lattices/end_tl2_7b_qf1sr_vf.mat', nturns)
runner.slurm.clean()
ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(ex), len(ey))).T

pickle.dump({'ex':ex, 'ey':ey, 'ie':ie}, open('data_EBS_emit_test.pkl', 'wb'))