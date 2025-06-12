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
ring = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/injection_nlk_mb.mat', mat_key='ring', check=False)
emit_x, emit_y = 30e-9,60e-9
e_spread, blength = 1.2e-3, 20e-3

dr_l = 0.0941
l_nlk = 0.3
nlk_latt = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/nlk_latt.mat', check=False)
rev_nlk_latt = nlk_latt.reverse(copy=True)

orbit_rev = rev_nlk_latt.find_orbit(refpts=at.End, orbit=np.array([-8.0e-3,0,0,0,0,0]))
orbit_rev[1][0][1] *= -1

nparts=1001
beta_opt = 7.0
betay = 2.04 #to check
betax_1 = beta_opt + ((0.5*l_nlk+dr_l+l_nlk)**2/beta_opt)
betay_1 = betay + ((0.5*l_nlk+dr_l+l_nlk)**2/betay)
alphax_1 = -(0.5*l_nlk+dr_l+l_nlk)/beta_opt
alphay_1 = -(0.5*l_nlk+dr_l+l_nlk)/betay

sigma_mat = at.sigma_matrix(emity=emit_y, emitx=emit_x, blength=blength, espread=e_spread,
                                alphay=alphay_1, alphax=alphax_1, betay=betay_1, betax=betax_1)
opt_beam = at.beam(nparts, sigma_mat, orbit=orbit_rev[1][0])

s_marker = l_nlk + dr_l +0.5*l_nlk
t_ring = ring.sbreak(break_s=s_marker)
idx_end3nlk = t_ring.get_uint32_index('sbreak')[0]
rot_ring = ring.rotate(idx_end3nlk)
idx_id04 = rot_ring.get_uint32_index('ID04')[0]
o6,*_ = rot_ring.find_orbit()

nturns = 1001
x = np.linspace(-3e-3,3e-3,15)
xp = np.linspace(-1e-3,1e-3,15)
all_beams_errors = []
for i in range(len(x)):
    print(i)
    for j in range(len(xp)):
        print(j)
        bi = copy.deepcopy(opt_beam)
        bi[0,:] += x[i]
        bi[1,:] += xp[j]
        _ = nlk_latt.track(bi,nturns=1,in_place=True, use_mp=True)
        for k in range(6):
            bi[k] += o6[k]
        _ = rot_ring.track(bi, nturns=1, refpts=idx_id04, in_place=True, use_mp=True)
        all_beams_errors.append(bi)
# exit()

runner = runners.Injeff_runner_EBS_thin(all_beams_errors, '/machfs/sauret/SharedCodes/tracking/lattices/injection_nlk_mb.mat', nturns)
runner.slurm.clean()
ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(x), len(xp))).T

pickle.dump({'x':x, 'xp':xp, 'ie':ie}, open('/machfs/sauret/SharedCodes/tracking/data_sim/data_EBS_only_nlk4.pkl', 'wb'))