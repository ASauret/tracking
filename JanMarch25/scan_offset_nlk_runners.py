import numpy as np
import at
import at.plot
import matplotlib.pyplot as plt
import scipy.constants as cst
import copy
import nlk
import pickle
import matplotlib
import runners
"""
Injeff scan wrt to the NLK peak field position
"""

def inj_eff(beam):
    lost = np.sum(np.isnan(beam[0, :, 0, -1]))
    eff = (np.shape(beam)[1] - lost) / np.shape(beam)[1]
    return eff * 100, lost

def get_coeff(field , xvals, degree):
    coeff = np.polyfit(xvals-xvals.mean(), field, degree, rcond=2e-18)
    fit = np.poly1d(coeff)
    return coeff, fit


Icoil = 5.5e3
x1_c = 3.7e-3
x2_c = 4.25e-3
y1_c = 11.7e-3
y2_c = 5.5e-3

icoil = -6.9e3
currents = np.array([-1, 1, -1, 1, -1, 1, -1, 1])*icoil
x1, x2 = 11, 6.2
y1, y2 = 11, 6.2
l_nlk = 0.3
brho = 20.16
ypos = np.array([y1, y2, y1, y2, -y1, -y2, -y1, -y2])*1.0e-3
xpos = np.array([x1, x2, -x1, -x2, x1, x2, -x1, -x2])*1.0e-3
kicker = nlk.Kicker(xpos, ypos, currents)
kicker.plot_kick(np.linspace(-20e-3,20e-3,1001),brho,l_nlk, show=True)
pn = kicker.get_polynoms(20e-3, 1001, 30)
nlk1 = kicker.gen_at_elem('NLK1', l_nlk, pn/brho)
nlk2 = kicker.gen_at_elem('NLK2', l_nlk, pn/brho)
nlk3 = kicker.gen_at_elem('NLK3', l_nlk, pn/brho)
dr_l = 0.0941
dr_nlk = at.Drift('DR_NLK',dr_l)

#Lattice NLK 
nlk_latt = at.Lattice([nlk1, dr_nlk, nlk2, dr_nlk, nlk3], name='NLK_LATT', energy=6e9)
at.save_lattice('/machfs/sauret/tracking/quad_effect/nlk_latt.mat')
rev_nlk_latt = nlk_latt.reverse(copy=True)
_,orbit_rev = rev_nlk_latt.find_orbit(refpts=at.End, orbit=np.array([-8.0e-3,0,0,0,0,0]))
orbit_rev[0][1] *= -1
print(orbit_rev[0])

# SR parameters
lattice = '/machfs/sauret/tracking/injection_nlk_mb.mat'
key = 'ring'
ring = at.load_lattice(lattice, mat_key=key)  # ring with injection section
[inj_opt, _, _] = at.get_optics(ring)
betax, betay = inj_opt.beta[0], inj_opt.beta[1]
alphax, alphay = inj_opt.alpha[0], inj_opt.alpha[1]
emity, emitx = 42e-9, 21e-9
ring.rf_voltage = 5.5e6
ring.enable_6d(cavity_pass='RFCavityPass')  # synchrotron radiation and rfc
orbit, _ = ring.find_orbit()
envelope = at.envelope_parameters(ring)
sigma_mat = at.sigma_matrix(emity=emity, emitx=emitx, blength=20e-3, espread=1.2e-3,
                            alphay=alphay, alphax=alphax, betay=betay, betax=7.0)

# coeff, fit = get_coeff(rescale, rescale_x, 30)
# poly_a = np.zeros(len(coeff))
# nlk_chams = at.ThinMultipole('NLK', length=0.0, poly_a=poly_a, poly_b=-np.flip(coeff))
# nlk_thin = pickle.load(open('/machfs/sauret/tracking/thin_nlk_elem.pickle', 'rb'))

p_in = np.zeros((6,1001))
amplitude = np.linspace(-15.0e-3, -5e-3, 1001)
p_in[0,:] =  amplitude
p_in[1,:] = 0
p_out, *_ = nlk_latt.track(p_in)
p_out = np.squeeze(p_out[:,:,0,0])
coeff, fit = get_coeff(p_out[1,:], amplitude, 30)

matplotlib.rcParams.update({'font.size': 12})
plt.plot(p_in[0,:], p_out[1,:], label='angle')
plt.plot(p_in[0,:],p_out[0,:], label='position')
plt.xlabel('x entry [m]')
plt.ylabel('x exit [m]')
# plt.plot(amplitude, fit(amplitude))
plt.legend()
plt.tight_layout()
plt.show()
# exit()
nparts = 101
nturns = 101

offset_scan = np.linspace(-0.003, 0.003, 15)
xp = np.linspace(-1e-3,1e-3,15)
eff_case = np.zeros(len(offset_scan))

s_marker = nlk3.Length + dr_l +0.5*nlk2.Length
t_ring = ring.sbreak(break_s=s_marker)
idx_end3nlk = t_ring.get_uint32_index('sbreak')[0]
rot_ring = ring.rotate(idx_end3nlk)
idx_id04 = rot_ring.get_uint32_index('ID04')[0]
o6,*_ = rot_ring.find_orbit()
beta_opt = 6.05
betax_1 = beta_opt + ((0.5*nlk2.Length+dr_l+nlk1.Length)**2/beta_opt)
betay_1 = betay + ((0.5*nlk2.Length+dr_l+nlk1.Length)**2/betay)
alphax_1 = -(0.5*l_nlk+dr_l+l_nlk)/beta_opt
alphay_1 = -(0.5*l_nlk+dr_l+l_nlk)/betay

sigma_mat_inj = at.sigma_matrix(emity=emity, emitx=emitx, blength=20e-3, espread=1.2e-3,
                                alphay=alphay_1, alphax=alphax_1, betay=betay_1, betax=betax_1)
inj_beam = at.beam(nparts, sigma_mat_inj,orbit_rev)
all_beam = []
for i in range(len(offset_scan)):
    print(offset_scan[i])
    for k in range(len(xp)):
        inj_beam_copy = copy.deepcopy(inj_beam)
        inj_beam_copy[0,:] += offset_scan[i]
        print(xp[k])
        inj_beam_copy[1,:] += xp[k]
        beam_out,*_ = nlk_latt.track(inj_beam_copy)
        beam_out = np.squeeze(beam_out[:,:,0,0])
        for j in range(6):
            beam_out[j] += o6[j]
        beam_out_r,*_ = rot_ring.track(beam_out, nturns=1, refpts=idx_id04, in_place=True, use_mp=True)
        beam_out_r = np.squeeze(beam_out_r[:,:,0,0])
        all_beam.append(beam_out_r)

runner = runners.Injeff_runner_EBS_thin(all_beam, '/machfs/sauret/SharedCodes/tracking/lattices/injection_nlk_mb.mat', nturns)
runner.slurm.clean()
ie = np.array(runner.submit())
runner.slurm.clean()

ie = ie.reshape((len(offset_scan), len(xp))).T
# ie = ie.reshape((len(offset_scan))).T

# pickle.dump({'x':offset_scan, 'ie':ie}, open('/machfs/sauret/SharedCodes/tracking/data_sim/booster_upgrade/data_EBS_2d_runners.pkl', 'wb'))
pickle.dump({'x':offset_scan, 'xp':xp, 'ie':ie}, open('/machfs/sauret/SharedCodes/tracking/data_sim/booster_upgrade/data_EBS_2d_runners.pkl', 'wb'))