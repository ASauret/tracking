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

with open('beam_opt.pickle', 'rb') as f:
    beam = pickle.load(f)

beam = np.squeeze(beam[:,:,0,0])
nturns = 2000
x = np.arange(-2e-3,2.25e-3,0.25e-3)
xp = np.arange(-2e-3,2.25e-3,0.25e-3)
xg,xpg = np.meshgrid(x,xp)

all_beams = []

for i in range(len(x)):
    for j in range(len(xp)):
        bi = copy.deepcopy(beam)
        bi[0] += x[i]
        bi[1] += xp[j]
        all_beams.append(bi)

runner = runners.Injeff_runner_EBS(all_beams, './ring_mik.mat', './end_tl2.pickle', 2000)
runner.slurm.clean()
ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(x), len(xp))).T

pickle.dump({'x':x, 'xp':xp, 'ie':ie}, open('data.pkl', 'wb'))



