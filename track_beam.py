import at
import numpy as np
import matplotlib.pyplot as plt
import copy
import runners
import pickle

ring = at.load_lattice('./ring_mik.mat')
ring.enable_6d()

idx_mik = ring.get_uint32_index('MIK')
idx_inj = ring.get_uint32_index('MIKEND')[0]

ring_nomik = ring.deepcopy()
for idx in idx_mik:
    ring_nomik[idx].PolynomB *= 0.0
    ring_nomik[idx].PolynomA *= 0.0
o0, _ = ring_nomik.find_orbit()

line_mik = ring[:idx_inj+1]
line_mik_inv = line_mik.reverse(copy=True)

p0 = np.array([-3.3e-3, 0.0, 0.0, 0.0, 0.0, 0.0])

_, o6 = line_mik_inv.find_orbit(refpts=at.End, orbit=p0)
oin = o6[0]
oin[1] *= -1
oin[3] *= -1
oin[5] *= 0.0

twiin = {'beta': [6.0, 6.0],
         'alpha': [0.0, 0.0],
         'dispersion': [0.0, 0.0, 0.0, 0.0]}


#x = np.linspace(-1.5e-3, 2.5e-3, 31)
#xp = np.linspace(-1.2e-3, 0.5e-3, 31)
x = np.linspace(-2.0e-3, 2.0e-3, 31)
xp = np.linspace(-1.0e-3, 1.0e-3, 31)


all_beams = []

beam = at.beam(1000, at.sigma_matrix(twiss_in=twiin,
                                   emitx=5.2e-9, emity=0.5e-9, 
                                   espread=0.96e-3, blength=7.5e-3),
               orbit=oin+o0)

for i, ix in enumerate(x):
    for ii, ixp in enumerate(xp):
        bi = copy.deepcopy(beam)
        bi[2] += ix
        bi[3] += ixp
        all_beams.append(bi)

runner = runners.Injeff_runner(all_beams, './ring_mik.mat', 1024, pulsed_elements_refpts=idx_mik)
runner.slurm.clean()
ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(x), len(xp))).T

pickle.dump({'x':x, 'xp':xp, 'ie':ie}, open('data.pkl', 'wb'))

fix, ax = plt.subplots()
ax.contourf(x, xp, ie, 50)
CS = ax.contour(x, xp, ie, levels=[0.5, 0.7, 0.9])
ax.clabel(CS, colors='black')
ax.set_xlabel('y [m]')
ax.set_ylabel("y' [rad]")
plt.show()
        
