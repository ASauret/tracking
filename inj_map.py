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

def track_idx(beam,ringpath,endtl2_path):
    ring = at.load_lattice(ringpath, mat_key='ring')
    with open(endtl2_path, 'rb') as f1:
        end_tl2 = pickle.load(f1)
    idx_nlk2 = end_tl2.get_uint32_index('NLK2')[0]
    idx_dr2 = end_tl2.get_uint32_index('DR_NLK2')[0]
    idx_nlk3 = end_tl2.get_uint32_index('NLK3')[0]
    s_marker = 0.5*end_tl2[idx_nlk2].Length + end_tl2[idx_dr2].Length + end_tl2[idx_nlk3].Length
    ring.enable_6d()
    new_ring = ring.sbreak(break_s=s_marker)
    idx_mk = new_ring.get_uint32_index('sbreak')[0]
    rot_ring = new_ring.rotate(idx_mk)
    idx_id04 = rot_ring.get_uint32_index('ID04')[0]
    return rot_ring.track(beam, nturns=1, refpts=idx_id04)

with open('beam_opt2.pickle', 'rb') as f:
    beam = pickle.load(f)

beam = np.squeeze(beam[:,:,0,0])
nturns = 2000
x = np.arange(-2e-3,2.25e-3,0.25e-3)
xp = np.arange(-2e-3,2.25e-3,0.25e-3)
xg,xpg = np.meshgrid(x,xp)

all_beams_errors = []
all_beams_ft = []

for i in range(len(x)):
    for j in range(len(xp)):
        bi = copy.deepcopy(beam)
        bi[0] += x[i]
        bi[1] += xp[j]
        all_beams_errors.append(bi)

#for k in range(len(all_beams_errors)):
#    all_beams_ft.append(track_idx(all_beams_errors[k], '/machfs/sauret/SharedCodes/tracking/injection_nlk_mb_qd3.mat', '/machfs/sauret/SharedCodes/tracking/end_tl2.pickle'))

#with open('/machfs/sauret/SharedCodes/tracking/beam_ft.pickle', 'wb') as ft:
#    pickle.dump(all_beams_ft, ft, pickle.HIGHEST_PROTOCOL)

runner = runners.Injeff_runner_EBS(all_beams_errors, '/machfs/sauret/SharedCodes/tracking/injection_nlk_mb_qd3.mat', 
                                   '/machfs/sauret/SharedCodes/tracking/end_tl2.mat', nturns)
runner.slurm.clean()
ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(x), len(xp))).T

pickle.dump({'x':x, 'xp':xp, 'ie':ie}, open('data_EBS.pkl', 'wb'))


