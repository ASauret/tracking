import at
from at.plot import plot_beta
import numpy as np
import nlk
import pickle
import matplotlib.pyplot as plt
import matplotlib
import runners
import copy
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

def phase_shifter(name, dmu, alpha, beta):
    m11 = np.cos(dmu) + alpha*np.sin(dmu)
    m12 =  beta*np.sin(dmu)
    m21 = (-1/beta)*(1+alpha**2)*np.sin(dmu)
    m22 = np.cos(dmu) - alpha*np.sin(dmu)
    full_mat = np.array([
    [m11, m12, 0, 0, 0, 0],
    [m21, m22, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
])
    m66 = at.M66(name,full_mat)
    return m66 

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
nlk1 = kicker.gen_at_elem('NLK1', length, pn/brho)
nlk2 = kicker.gen_at_elem('NLK2', length, pn/brho)
nlk3 = kicker.gen_at_elem('NLK3', length, pn/brho)
drift = at.Drift('DR_NLK', 0.0941)

tl2 = at.load_lattice('/machfs/sauret/injection/optics_matching/Lattices/June_25/18/TL2_beta5m.mat', check=False)
twiin = {'beta': [6.73,5.21],
         'alpha': [-2.21,1.4775],
         'ClosedOrbit': np.zeros(4),
         'dispersion': [-0.136, -0.092, 0, 0],
         'mu': [0, 0]
         }

idx_s3 = tl2.get_uint32_index('S3')[0]
_, _, l_opt = tl2.get_optics(refpts=idx_s3+1, twiss_in=twiin)
_,_, nlk_lopt = tl2.get_optics(refpts='NLK1', twiss_in=twiin)
phase_shift = phase_shifter('Shifter', np.pi, nlk_lopt.alpha[0], nlk_lopt.beta[0])

end_tl2 = tl2[idx_s3+1:]
idx_drnlk2 = end_tl2.get_uint32_index('DR_NLK2')[0]
idx_drnlk3 = end_tl2.get_uint32_index('DR_NLK3')[0]
idx_drqf1 = end_tl2.get_uint32_index('DR_QF1')[0]
idx_center = end_tl2.get_uint32_index('Center')[0]
new_length = 0.0941
new_length_drqf1 = 0.1254
end_tl2[idx_drnlk2].Length = new_length
end_tl2[idx_drnlk3].Length = new_length
end_tl2[idx_drqf1].Length = new_length_drqf1

idx_qf1 = end_tl2.get_uint32_index('QF1_SR')[0]
idx_nlk1 = end_tl2.get_uint32_index('NLK1')[0]
idx_nlk2 = end_tl2.get_uint32_index('NLK2_1')[0]
qf1_sr = at.Quadrupole('QF1_SR', end_tl2[idx_qf1].Length, bending_angle= -0.00507068, k=end_tl2[idx_qf1].K)
end_tl2 = end_tl2[:idx_qf1] + [qf1_sr] + end_tl2[idx_qf1+1:]
end_tl2 = end_tl2[:idx_nlk1] + [nlk1] + end_tl2[idx_nlk1+1:]
end_tl2 = end_tl2[:idx_nlk2] + [nlk2] + end_tl2[idx_nlk2+3:]
idx_nlk3 = end_tl2.get_uint32_index('NLK3')[0]
end_tl2 = end_tl2[:idx_nlk3] + [nlk3] + end_tl2[idx_nlk3+1:]

dx = np.linspace(-15.0e-3, 15.0e-3, 101)
dxo = []
for i, dxi in enumerate(dx):  
    dx6 = np.array([dxi, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    dxout, *_ = end_tl2.track(dx6)
    dxo.append(np.squeeze(dxout)[1])
plt.plot(dx, dxo)
plt.show()

injoff = -8e-3
rev_end_tl2 = end_tl2.reverse(copy=True)
orbit_rev = rev_end_tl2.find_orbit(refpts=at.End, orbit=np.array([injoff,0,0,0,0,0]))
print(orbit_rev[1][0][0])
orbit_rev[1][0][1] *= -1
orbit_rev[1][0][3] *= -1

ey=54e-9
ex=27e-9
bl=20e-3
ep=1.2e-3
ring = at.load_lattice('/machfs/sauret/injection/lattices/before_injury/injection_nlk_mb.mat')
ring.enable_6d(cavity_pass='RFCavityPass')
o6, _ = ring.find_orbit()
l0, _, _ = at.get_optics(ring.disable_6d(copy=True), method=at.linopt2)

nparts=1001
nturns = 1024
sextu_on = True
k2 = 100
sextu = at.ThinMultipole('SEXT', poly_a=np.zeros(4), poly_b=np.array([0.0,0.0,k2,0.0]))
sextu = at.Lattice([sextu], energy=6.0e9, periodicity=1)

x = np.linspace(-1e-3,1e-3,21)
xp = np.linspace(-0.5e-3,0.5e-3,21)
all_beams_errors = []

for i in range(len(x)):
    print(i)
    for j in range(len(xp)):
        print(j)
        err = np.array([x[i],xp[j], 0.0, 0.0, 0.0, 0.0])
        sigma_mat = at.sigma_matrix(twiss_in=l_opt[0] ,emity=ey, emitx=ex, blength=bl, espread=ep)                       
        beam = at.beam(nparts, sigma_mat, orbit=orbit_rev[1][0]+err)
        if sextu_on:
            beam[0,:] -= orbit_rev[1][0][0]
            sextu.track(beam, in_place=True)
            beam[0,:] += orbit_rev[1][0][0]
        # plt.scatter(beam[0],beam[1])
        # plt.show()
        # exit()
        bi,*_ = end_tl2.track(beam,nturns=1, use_mp=True)
        bi = np.squeeze(bi[:,:,0,0])
        all_beams_errors.append(bi)
print('Matrix size :', len(x)*len(xp))

runner = runners.Injeff_runner_EBS(all_beams_errors, '/machfs/sauret/injection/lattices/before_injury/injection_nlk_mb.mat', end_tl2, nturns)
runner.slurm.clean()

ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(x), len(xp))).T

pickle.dump({'x':x, 'xp':xp, 'ie':ie , 'nturns':nturns, 'npart':nparts}, open('/machfs/sauret/injection/results/EBS_booster_upgd/10p_sextu100_v2.pkl', 'wb'))
