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

lattice_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/newtl2_7b.mat', check=False)

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

end_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/end_tl2_7b_8mm.mat', check=False)
rev_end_tl2 = end_tl2.reverse(copy=True)


orbit_rev = rev_end_tl2.find_orbit(refpts=at.End, orbit=np.array([-8.0e-3,0,0,0,0,0]))
orbit_rev[1][0][1] *= -1

nparts=1001
nturns = 1001
sigma_mat = at.sigma_matrix(twiss_in=l_opt[0] ,emity=emit_y, emitx=emit_x, blength=blength, espread=e_spread)
opt_beam = at.beam(nparts, sigma_mat)

# sextu = at.ThinMultipole('SEXTU', poly_b=[0,0,-130], poly_a=[0,0,0])
# sextu.MaxOrder = 2
# opt_beam2 = sextu.track(opt_beam)

for k in range(6):
    opt_beam[k] += orbit_rev[1][0][k]

x = np.linspace(-3e-3,3e-3,2)
xp = np.linspace(-1e-3,1e-3,2)
all_beams_errors = []
for i in range(len(x)):
    print(i)
    for j in range(len(xp)):
        print(j)
        bi = copy.deepcopy(opt_beam)
        # bi[0,:] += x[i]
        # bi[1,:] += xp[j]
        _ = end_tl2.track(bi,nturns=1,in_place=True, use_mp=True)
        plt.scatter(bi[0],bi[1])
        plt.show()
        exit()
        all_beams_errors.append(bi)
# exit()

runner = runners.Injeff_runner_EBS(all_beams_errors, '/machfs/sauret/SharedCodes/tracking/lattices/injection_nlk_mb.mat', 
                                   '/machfs/sauret/SharedCodes/tracking/lattices/end_tl2_7b_qf1sr_vf.mat', nturns)
runner.slurm.clean()
ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(x), len(xp))).T

pickle.dump({'x':x, 'xp':xp, 'ie':ie}, open('/machfs/sauret/SharedCodes/tracking/data_sim/18_03_25/beta7_data_EBS_full_energy.pkl', 'wb'))


