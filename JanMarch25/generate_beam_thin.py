import at
import nlk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 15})

def get_coeff(field , xvals, degree):
    coeff = np.polyfit(xvals-xvals.mean(), field, degree, rcond=2e-18)
    fit = np.poly1d(coeff)
    return coeff, fit


lattice_tl2 = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/newtl2.mat', check=False)

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

nparts= 1001
beam = at.beam(nparts, at.sigma_matrix(twiss_in=l_opt[0], emity=emit_y, emitx=emit_x, espread=e_spread, blength=blength))
beam[0] -= 17.0e-3
beam[1] += 6.1e-3

end_tl2 = lattice_tl2[idx_s3+1:]
idx_drnlk2 = end_tl2.get_uint32_index('DR_NLK2')[0]
idx_drnlk3 = end_tl2.get_uint32_index('DR_NLK3')[0]
idx_drqf1 = end_tl2.get_uint32_index('DR_QF1')[0]
idx_center = end_tl2.get_uint32_index('Center')[0]
new_length = 0.0941
new_length_drqf1 = 0.1254
end_tl2[idx_drnlk2].Length = new_length
end_tl2[idx_drnlk3].Length = new_length
end_tl2[idx_drqf1].Length = new_length_drqf1

icoil = -6.9e3
currents = np.array([-1, 1, -1, 1, -1, 1, -1, 1])*icoil
x1, x2 = 12.8, 6.4
y1, y2 = 12.8, 6.4
l_nlk = 0.3
brho = 20.16
ypos = np.array([y1, y2, y1, y2, -y1, -y2, -y1, -y2])*1.0e-3
xpos = np.array([x1, x2, -x1, -x2, x1, x2, -x1, -x2])*1.0e-3
kicker = nlk.Kicker(xpos, ypos, currents)
pn = kicker.get_polynoms(20e-3, 1001, 30)
nlk1 = kicker.gen_at_elem('NLK1', l_nlk, pn/brho)
nlk2 = kicker.gen_at_elem('NLK2', l_nlk, pn/brho)
nlk3 = kicker.gen_at_elem('NLK3', l_nlk, pn/brho)
pn_thin = np.loadtxt('/machfs/sauret/SharedCodes/tracking/coeff.txt')
# print(pn_thin)
nlk_thin = kicker.gen_at_elem('ThinNLK', 0.0, np.flip(pn_thin))
# print(nlk_thin)


idx_qf1 = end_tl2.get_uint32_index('QF1_SR')[0]
idx_nlk1 = end_tl2.get_uint32_index('NLK1')[0]
idx_nlk2 = end_tl2.get_uint32_index('NLK2_1')[0]

qf1_sr = at.Quadrupole('QF1_SR', end_tl2[idx_qf1].Length, k=end_tl2[idx_qf1].K)
sextu = at.ThinMultipole('SEXTU', poly_b=[0,0,120], poly_a=[0,0,0])
sextu.MaxOrder = 2
end_tl2 = end_tl2[:idx_qf1] + [qf1_sr] + end_tl2[idx_qf1+1:]
end_tl2 = end_tl2[:idx_nlk1] + [at.Drift('DR_NLK', nlk1.Length)] + end_tl2[idx_nlk1+1:]
end_tl2 = end_tl2[:idx_center] + [nlk_thin] + end_tl2[idx_center+1:]
idx_thin = end_tl2.get_uint32_index('ThinNLK')[0]

end_tl2_thin = end_tl2[:idx_thin+1]
idx_nlk21 = end_tl2_thin.get_uint32_index('NLK2_1')[0]
end_tl2_thin[idx_nlk21] = at.Drift('DR_NLK', 0.15)


idx_nlk3 = end_tl2.get_uint32_index('NLK3')[0]
end_tl2 = end_tl2[:idx_nlk3] + [nlk3] + end_tl2[idx_nlk3+1:]

for e in end_tl2_thin:
    print(e)

pickle.dump(nlk_thin, open('/machfs/sauret/tracking/thin_nlk_elem.pickle', 'wb'))
exit()

p_in = np.zeros((6,1001))
amplitude = np.linspace(-15.0e-3, 15.0e-3, 1001)
p_in[0,:] =  amplitude
p_in[1,:] = 0
p_out = nlk_thin.track(p_in)
# p_out = np.squeeze(p_out[:,:,0,0])

coeff, fit = get_coeff(p_out[1,:], p_in[0,:], 30)
print(coeff)
np.savetxt('/machfs/sauret/SharedCodes/tracking/coeff.txt', coeff)

plt.plot(p_in[0,:], p_out[1,:])
plt.plot(p_in[0,:], fit(p_in[0,:]))
plt.xlabel('Particles amplitude [m]')
plt.ylabel('Deflection [rad]')
plt.show()
plt.tight_layout()
# at.save_lattice(end_tl2_thin,'/machfs/sauret/SharedCodes/tracking/lattices/end_tl2_thinNLK.mat')
# exit()


rev_end_tl2 = end_tl2.reverse(copy=True)

orbit_rev = rev_end_tl2.find_orbit(refpts=at.End, orbit=np.array([-8.0e-3,0,0,0,0,0]))
orbit_rev[1][0][1] *= -1
# print(orbit_rev)

# opt_beam = at.beam(1001, at.sigma_matrix(twiss_in=l_opt[0], emity=emit_y, emitx=emit_x, espread=e_spread, blength=blength),
#                    orbit=orbit_rev[1][0])


opt_beam = at.beam(1001, at.sigma_matrix(twiss_in=l_opt[0], emity=emit_y, emitx=emit_x, espread=e_spread, blength=blength))
opt_beam = sextu.track(opt_beam)
for i in range(6):
   opt_beam[i] += orbit_rev[1][0][i]


beam_out, *_ = end_tl2.track(opt_beam, use_mp=True)
beam_out = np.squeeze(beam_out[:,:,0,0])
plt.scatter(beam_out[0], beam_out[1])
plt.show()
exit()

beam_out_no, *_ = end_tl2.track(beam, use_mp=True)

# with open('/machfs/sauret/tracking/beam_opt.pickle', 'wb') as f:
#     pickle.dump(beam_out, f, pickle.HIGHEST_PROTOCOL)

# beam_out = np.squeeze(beam_out[:,:,0,0])
beam_out_no = np.squeeze(beam_out_no[:,:,0,0])
plt.plot(opt_beam[0,:], opt_beam[1,:], '.', label='Beam in - optimized')
plt.plot(beam_out[0,:], beam_out[1,:], '.', label='Beam out - optimized')
plt.plot(beam[0,:], beam[1,:], '.', label='Beam in')
plt.plot(beam_out_no[0,:], beam_out_no[1,:], '.', label='Beam out')
plt.legend(fontsize=10)
plt.xlabel(r'x [m]')
plt.ylabel(r'x [mrad]')
plt.tight_layout()
plt.show()

