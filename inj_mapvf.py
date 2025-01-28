import at
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import copy
import runners
matplotlib.rcParams.update({'font.size': 15})

x = np.arange(-2e-3,2.25e-3,0.25e-3)
xp = np.arange(-2e-3,2.25e-3,0.25e-3)

#Previously tracked beam between end of NLK3 and ID04
with open('/machfs/sauret/SharedCodes/tracking/beam_ft.pickle', 'rb') as ft:
    all_beams_ft = pickle.load(ft)

runner = runners.Injeff_runner(all_beams_ft, '/machfs/sauret/SharedCodes/tracking/injection_nlk_mb_qd3.mat', 2000)
runner.slurm.clean()
ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(x), len(xp))).T

pickle.dump({'x':x, 'xp':xp, 'ie':ie}, open('data_EBS.pkl', 'wb'))