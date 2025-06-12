import at
import nlk
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 15})
import matplotlib.pyplot as plt

icoil = -6.9e3
nlk_scale1 = 0.95
nlk_scale2 = 0.95
ksext = 0
entry = -17.0e-3
exit = -8.0e-3
angle = 6.1e-3

brho = 20.16
length = 0.3
currents = np.array([-1, 1, -1, 1, -1, 1, -1, 1])*icoil
y1 = 14*nlk_scale1
x1 = 14*nlk_scale1
y2 = 7*nlk_scale2
x2 = 7*nlk_scale2
print(x1, y1, x2, y2)
ypos = np.array([y1, y2, y1, y2, -y1, -y2, -y1, -y2])*1.0e-3
xpos = np.array([x1, x2, -x1, -x2, x1, x2, -x1, -x2])*1.0e-3
    
kicker = nlk.Kicker(xpos, ypos, currents)
#kicker.plot_kick(np.linspace(-20.0e-3, 20.0e-3, 1001), brho, length, show=False)

pn = kicker.get_polynoms(20e-3, 1001, 30)
nlk1 = kicker.gen_at_elem('NLK1', length, pn/brho)
nlk2 = kicker.gen_at_elem('NLK2', length, pn/brho)
nlk3 = kicker.gen_at_elem('NLK3', length, pn/brho)
dr_nlk = at.Drift('DR_NLK', 13.0e-2/2)
dr_qf1 = at.Drift('DR_NLK', 13.0e-2 + 0.049068/2)
dr_s3 = at.Drift('DR_NLK', 0.049068/2)

ring = at.load_lattice('./injection_nlk_mb_qd3.mat')
ring.disable_6d()
qf1 = ring.get_elements(at.Quadrupole)[-1]
sext = at.ThinMultipole('SEXT', [0, 0, 0], poly_b=[0, 0, ksext])
at.set_shift([sext], dxs=entry, dzs=0.0)

lat = at.Lattice([dr_s3, sext, qf1, dr_qf1, nlk1, dr_nlk, nlk2, dr_nlk, nlk3], energy=6.0e9, periodicity=1)
nparts = 1001
x = np.linspace(-20e-3, -5.0e-3, nparts)
part = np.zeros((6, nparts))
part[0, :] = x[:]
part[1, :] = angle
lat.track(part, in_place=True)

delta = 10.0e-6
pin = np.array([entry, angle, 0.0, 0.0, 0.0, 0.0])
pinp = np.array([entry+delta, angle, 0.0, 0.0, 0.0, 0.0])
pinm = np.array([entry-delta, angle, 0.0, 0.0, 0.0, 0.0])
pout, *_ = lat.track(pin, refpts=at.All)
poutp, *_ = lat.track(pinp, refpts=at.All)
poutm, *_ = lat.track(pinm, refpts=at.All)
pout= np.squeeze(pout)
poutp = np.squeeze(poutp)
poutm = np.squeeze(poutm)
lelem = np.array([e.Length for e in lat])

offsets = pout[0,:]
k0l = np.diff(pout[1,:])
k1 = (np.diff(poutp[1,:]-poutm[1,:])/2/delta)/lelem
print(offsets)
print(k0l)
print(k1)

x_k = np.linspace(-18.0e-3, 0, 101)
k_qf1 = x_k*k1[2]*lelem[2]
plt.figure()
plt.plot(x_k, k_qf1, label='Deflection', linewidth=3)
plt.plot(offsets[2], k1[2]*offsets[2]*lelem[2], '.', markersize=20, label='TL2 beam')
plt.xlabel('x [m]')
plt.ylabel(r"$\Delta$x' [rad]")
plt.title(r"$\Delta$x="+str(np.round(offsets[2], 4))+
          ", $\Delta$x'="+str(np.round(k0l[2], 4))+
          ", $k_1$L="+str(np.round(k1[2]*lelem[2], 4)))
plt.legend()
plt.tight_layout()

_, k_nlk1 = kicker.get_kick(x_k, 0, brho, lelem[4])
_, k_nlk1o = kicker.get_kick([offsets[4]], 0, brho, lelem[4])
plt.figure()
plt.plot(x_k, k_nlk1, label='Deflection', linewidth=3)
plt.plot(offsets[4], k_nlk1o, '.', markersize=20, label='TL2 beam')
plt.xlabel('x [m]')
plt.ylabel(r"$\Delta$x' [rad]")
plt.title(r"$\Delta$x="+str(np.round(offsets[4], 4))+
          ", $\Delta$x'="+str(np.round(k0l[4], 4))+
          ", $k_1$L="+str(np.round(k1[4]*lelem[4], 4)))
plt.legend()
plt.tight_layout()

_, k_nlk2 = kicker.get_kick(x_k, 0, brho, lelem[6])
_, k_nlk2o = kicker.get_kick([offsets[6]], 0, brho, lelem[6])
plt.figure()
plt.plot(x_k, k_nlk2, label='Deflection', linewidth=3)
plt.plot(offsets[6], k_nlk2o, '.', markersize=20, label='TL2 beam')
plt.xlabel('x [m]')
plt.ylabel(r"$\Delta$x' [rad]")
plt.title(r"$\Delta$x="+str(np.round(offsets[6], 4))+
          ", $\Delta$x'="+str(np.round(k0l[6], 4))+
          ", $k_1$L="+str(np.round(k1[6]*lelem[6], 4)))
plt.legend()
plt.tight_layout()

_, k_nlk3 = kicker.get_kick(x_k, 0, brho, lelem[8])
_, k_nlk3o = kicker.get_kick([offsets[8]], 0, brho, lelem[8])
plt.figure()
plt.plot(x_k, k_nlk3, label='Deflection', linewidth=3)
plt.plot(offsets[8], k_nlk3o, '.', markersize=20, label='TL2 beam')
plt.xlabel('x [m]')
plt.ylabel(r"$\Delta$x' [rad]")
plt.title(r"$\Delta$x="+str(np.round(offsets[8], 4))+
          ", $\Delta$x'="+str(np.round(k0l[8], 4))+
          ", $k_1$L="+str(np.round(k1[8]*lelem[8], 4)))
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(x, part[1,:], label='angle', linewidth=3)
plt.plot(x, part[0,:], label='position', linewidth=3)
plt.axvline(x=entry, color='k', linestyle='dashed')
plt.axhline(y=exit, color='k', linestyle='dashed')
plt.axhline(y=0.0, color='k', linestyle='dashed')
#plt.plot(x, x+(exit-entry), color='k', linestyle='dashed')
plt.xlabel('x entry [m]')
plt.ylabel('x exit [m]')
plt.legend()
plt.tight_layout()


plt.show()
