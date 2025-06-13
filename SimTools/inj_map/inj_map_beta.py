import at
import numpy as np
import nlk
import pickle
import matplotlib.pyplot as plt
import matplotlib
import runners
matplotlib.rcParams.update({'font.size': 15})

def inj_eff(beam):
    beam0 = beam[0]
    lost = np.sum(np.isnan(beam0[0, :, 0, -1]))
    eff = (np.shape(beam0)[1] - lost) / np.shape(beam0)[1]
    return eff * 100, lost

def get_polynoms(xf, yf, radius, npoints, norder):
    x = np.linspace(-radius, radius, npoints)
    y = np.interp(x, xf, yf)
    pn = np.polyfit(x, y, norder)
    return np.flip(pn)

brho = 20.16
length = 0.3
icoil = -6.9e3
currents = np.array([-1, 1, -1, 1, -1, 1, -1, 1])*icoil
y1 = 12
x1 = 12
y2 = 6
x2 = 6
print(x1, y1, x2, y2)
ypos = np.array([y1, y2, y1, y2, -y1, -y2, -y1, -y2])*1.0e-3
xpos = np.array([x1, x2, -x1, -x2, x1, x2, -x1, -x2])*1.0e-3

kicker = nlk.Kicker(xpos, ypos, currents)
pn = kicker.get_polynoms(20e-3, 1001, 30)
nlk = kicker.gen_at_elem('NLK', length, pn/brho)
drift = at.Drift('DR_NLK', 0.0941)

nlks = at.Lattice([nlk, drift, nlk, drift, nlk], energy=6.0e9, periodicity=1)

dx = np.linspace(-15.0e-3, 15.0e-3, 101)
dxo = []
for i, dxi in enumerate(dx):  
    dx6 = np.array([dxi, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    dxout, *_ = nlks.track(dx6)
    dxo.append(np.squeeze(dxout)[1])
plt.plot(dx, dxo)
plt.show()


pn = get_polynoms(dx, dxo, 15.0e-3, 101, 31)
elem = at.ThinMultipole('NLK_INT', np.zeros(len(pn)), pn*(-1))
nlks2 = at.Lattice([elem], energy=6.0e9, periodicity=1)

pn_fun = np.poly1d(np.flip(pn))
xp = pn_fun(8e-3)

ey=60e-9
ex=30e-9
bl=20e-3
ep=1.2e-3
ring = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/injection_nlk_mb.mat')
ring.enable_6d()
o6, _ = ring.find_orbit()
l0, _, _ = at.get_optics(ring.disable_6d(copy=True), method=at.linopt2)
emit_x, emit_y = 30e-9,60e-9
e_spread, blength = 1.2e-3, 20e-3

nparts=1001
nturns = 1001

x = np.linspace(-3e-3,3e-3,15)
beta = np.linspace(1.0,15.0, 15)
all_beams_errors = []

for i in range(len(x)):
    print(i)
    for j in range(len(beta)):
        print(j)
        dx6 = np.array([8e-3, -xp, 0.0, 0.0, 0.0, 0.0]) 
        sigma_mat = at.sigma_matrix(betax=beta[j], betay=l0.beta[1],
                            alphax=l0.alpha[0], alphay=l0.alpha[1],
                            emity=ey, emitx=ex,
                            blength=bl, espread=ep)                        
        beam = at.beam(nparts, sigma_mat, orbit=o6+dx6)
        beam[1,:] += x[i]
        bi,*_ = nlks2.track(beam,nturns=1, use_mp=True)
        bi = np.squeeze(bi[:,:,0,0])
        # plt.scatter(bi[0],bi[1])
        # plt.show()
        # exit()
        all_beams_errors.append(bi)
print('Matrix size :', len(x)*len(beta))

runner = runners.Injeff_runner_EBS_thin(all_beams_errors, '/machfs/sauret/SharedCodes/tracking/lattices/injection_nlk_mb.mat', nturns)
runner.slurm.clean()

ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(x), len(beta))).T

pickle.dump({'xp':x, 'b':beta, 'ie':ie}, open('/machfs/sauret/injection/results/beta_scan_xp.pkl', 'wb'))


