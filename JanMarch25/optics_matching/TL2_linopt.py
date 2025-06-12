import at
from at.plot import plot_beta
import matplotlib.pyplot as plt
import numpy
from at.latticetools import LocalOpticsObservable, GlobalOpticsObservable
from at.latticetools import EmittanceObservable, ObservableList
from at.future import RefptsVariable, VariableList, match
"""
Initial matching
"""

def max_beta(elemdata):
    # Evaluation function for the phase advance (both planes)
    beta = elemdata.beta
    return numpy.amax(beta)


def phase_advance(elemdata):
    mu = elemdata.mu
    dmu = (mu[1] - mu[0]) / 2 / numpy.pi
    return numpy.mod(dmu, 1)[0]


def get_b_db(filename, quad, offset):
    data = numpy.loadtxt(filename, skiprows=3)
    x = data[:, 0] * 1.0e-3
    y = data[:, 1] * 1.0e-3 / 20.16
    dy = numpy.gradient(y, x)
    fact = dy[0] / quad.K
    dy /= fact
    y /= fact
    b = numpy.interp(offset, x, y)
    db = numpy.interp(offset, x, dy)
    return b, db



def move_elem(ring, name, ds):
    idx_elem = ring.get_uint32_index(name)[0]
    idx_db = idx_elem
    idx_da = idx_elem
    while not isinstance(ring[idx_da], at.Drift):
        idx_da += 1      
    while not isinstance(ring[idx_db], at.Drift):
        idx_db -= 1    
    if (ring[idx_db].Length < -ds) or (ring[idx_da].Length < ds):
        raise ValueError('Displacement larger than drift')
    ring[idx_db].Length += ds
    ring[idx_da].Length -= ds

    
#storage ring
ring = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/injection_nlk_mb_qd3.mat')
ring.disable_6d()
l0, *_ = ring.linopt2()

qf1sr = ring.get_elements(at.Quadrupole)[-1]
qd2sr = ring.get_elements(at.Quadrupole)[-2]

b_qf1, db_qf1 = get_b_db('/machfs/sauret/SharedCodes/tracking/optics_matching/qd2_field_int.dat', qf1sr, 0.0169)
b_qd2, db_qd2 = get_b_db('/machfs/sauret/SharedCodes/tracking/optics_matching/qd2_field_int.dat', qd2sr, 0.16)

#  tl2
#  Theoretical off-momentum values with Qx=12.75
twiin = {'beta': [6.73,5.21],
         'alpha': [-2.21,1.4775],
         'ClosedOrbit': numpy.zeros(4),
         'dispersion': [-0.136, -0.092, 0, 0],
         'mu': [0, 0]
         }


#  Measured values from 2018 transported to SE2
#twiin = {'beta': [5.90, 5.65],
#         'alpha': [-1.71, 1.47],
#         'ClosedOrbit': numpy.zeros(4),
#         'dispersion': [-0.145, -0.092, 0, 0],
#         'mu': [0, 0]
#         }

tl = at.load_lattice('/machfs/sauret/SharedCodes/tracking/lattices/present_TL2.mat', use='betamodel', periodicity=1)
tlinv = tl.reverse(copy=True)
idx_d5 = tlinv.get_uint32_index('D5')[0]
#tl.plot_beta(twiss_in=twiin)
#plt.show()

#QF1 INJ
qf1 = at.Dipole('QF1_SR', qf1sr.Length, bending_angle=-0.00507068, k=1.74129177*(0))

#QD2 INJ
qd2 = at.Dipole('QD2_SR', qd2sr.Length, bending_angle=-b_qd2*qd2sr.Length, k=db_qd2)
dr_qd2 = 0.24936

#S3
s3 = at.Dipole('S3', 0.8, bending_angle=50e-3)
dr_s3 = 0.0723 + 0.049068/2

#S2
s2 = at.Dipole('S2', 0.8, bending_angle=35e-3)
dr_s2 = 20.0e-2

#S1
s1 = at.Dipole('S1', 0.8, bending_angle=35e-3)
dr_s1 = 0.22139


#Additional TL2 quads
lq = 0.23 #previous 0.23
q13_tl2 = qf1sr.deepcopy()
q13_tl2.FamName = 'QF13'
q13_tl2.Length = lq
q12_tl2 = qf1sr.deepcopy()
q12_tl2.FamName = 'QD12'
q12_tl2.Length = lq

#NLK params - form tracking
nnlk = 3
dr_nlk = 13.0e-2/2
lnlk = 0.3
k0_nlk3 = 0.00331618
k0_nlk2 = 0.00384397
k0_nlk1 = 0.003945
k1_nlk3 = -1.19257747*0
k1_nlk2 = -0.7139671*0
k1_nlk1 = 0.4535935*0

tl2 = [at.Marker('End')]
tl2 += [at.Dipole('NLK3', lnlk, k0_nlk3, k=k1_nlk3)]
tl2 += [at.Drift('DR_NLK2', dr_nlk), at.Dipole('NLK2_1', lnlk/2.0, k0_nlk2/2.0, k=k1_nlk2/2),
        at.Marker('Center'), at.Dipole('NLK2_1', lnlk/2.0, k0_nlk2/2.0, k=k1_nlk2/2)]
tl2 += [at.Drift('DR_NLK3', dr_nlk), at.Dipole('NLK1', lnlk, k0_nlk1, k=k1_nlk1)]
tl2 += [at.Drift('DR_QF1', 2*dr_nlk + 0.049068/2), qf1]
tl2 += [at.Drift('DR_S3', dr_s3), s3]
tl2 += [at.Drift('DR_S2', dr_s2), s2]
tl2 += [at.Drift('DR_QD2', dr_qd2), qd2]
tl2 += [at.Drift('DR_S1', dr_s1), s1]
tl2 += [at.Drift('DR_QF13', 0.77),
        q13_tl2,
        at.Drift('DR_QD12', 0.94),
        q12_tl2,
        at.Drift('DR_D5', 2.124733)
        ]
tl2 += [at.Marker('D5_End')]
tl2 += tlinv[idx_d5:]
tl2 = at.Lattice(tl2, energy=6e9, periodicity=1)
tl2 = tl2.reverse(copy=True)

d5idx = tl2.get_uint32_index('D5')[0]
tl2[d5idx-1].Length = 1.03998
print(tl2[d5idx-1].FamName, tl2[d5idx-1].Length)


#qf72 = tl2.get_elements('QF7')[0].deepcopy()
#qf72.FamName = 'QF7_2'
#dr23 = tl2.get_elements('DR_23')[0].deepcopy().insert(((0.5, qf72),))
#idx_dr23 = tl2.get_uint32_index('DR_23')[0]
#tl2 = tl2[:idx_dr23] + [*dr23] + tl2[idx_dr23+1:]

scenter = tl2.get_s_pos('Center')[0]
spos = tl2.get_s_pos(refpts=at.All)
for s, e in zip(spos, tl2):
      print(s-scenter, e.FamName, e.Length)

variables = VariableList()
quads = tl2.get_elements(at.Quadrupole)
for q in quads:
    nm = q.FamName
    #bounds = (-1.6, 1.6)
    variables.append(RefptsVariable(nm, "PolynomB", 1, name=nm, ring=tl2))
#variables.append(RefptsVariable('NLK1', "PolynomB", 1, name='NLK1', ring=tl2))

#tl2['QF7_2'][0].K = -0.4
#tl2['QF7_2'][0].PassMethod = 'DriftPass'
tl2['QF9'][0].K = 0.0
tl2['QF9'][0].PassMethod = 'DriftPass'
#tl2['QD5'][0].K = 0.0
#tl2['QF11'][0].K += -0.4
#tl2['QD12'][0].K = -0.8
#tl2['QD12'][0].PassMethod = 'DriftPass'
tl2['QD12'][0].K = -1.6
#tl2['QD13'][0].PassMethod = 'DriftPass'
tl2['QF13'][0].K = 0.9
tl2['D5'][0].BendingAngle -= 15e-3
#move_elem(tl2, 'SL_QF6', -2.0)
#move_elem(tl2, 'QF6', -2.0)
#move_elem(tl2, 'QF7', 0.0)
#move_elem(tl2, 'D4', -1.0)
#move_elem(tl2, 'SL_QD8', -1.0)
#move_elem(tl2, 'QD8', -1.0)
move_elem(tl2, 'QD12', 0.1)
move_elem(tl2, 'QF13', 0.0)



for e in tl2:
    print(e.FamName, e.Length, e.PassMethod)

l0.beta[0] = 7.0
l0.alpha = [0.0, 0.0]

idx_sext = tl2.get_uint32_index('SEXT')[0]
idx_center = tl2.get_uint32_index('Center')[0]
match_pt = 'End'

obs = ObservableList(twiss_in=twiin, method=at.linopt2)
obs.append(LocalOpticsObservable(match_pt, "beta", plane=0, target=l0['beta'][0]))
obs.append(LocalOpticsObservable(match_pt, "beta", plane=1, target=l0['beta'][1]))
obs.append(LocalOpticsObservable(match_pt, "alpha", plane=0, target=l0['alpha'][0]))
obs.append(LocalOpticsObservable(match_pt, "alpha", plane=1, target=l0['alpha'][1]))
obs.append(LocalOpticsObservable(match_pt, "dispersion", plane=0, target=l0['dispersion'][0]))
obs.append(LocalOpticsObservable(match_pt, "dispersion", plane=1, target=l0['dispersion'][1]))
obs.append(LocalOpticsObservable('SEXT', "dispersion", plane=0, target=0.0))
# obs.append(LocalOpticsObservable('SEXT', "dispersion", plane=1, target=0.0, bounds=(-1e-4,1e-4)))
obs.append(LocalOpticsObservable('SEXT', "alpha", plane=0, target=0.0))
#obs.append(LocalOpticsObservable('SEXT', "beta", plane=0, target=40.0, bounds=(0.0, numpy.inf)))
obs.append(LocalOpticsObservable(at.All, "beta", plane=1, target=100.0, bounds=(-numpy.inf, 0.0), statfun=numpy.amax))
obs.append(LocalOpticsObservable(at.All, "beta", plane=0, target=110.0, bounds=(-numpy.inf, 0.0), statfun=numpy.amax))
#obs.append(LocalOpticsObservable(at.All, "dispersion", plane=0, target=1.5, bounds=(-numpy.inf, 0.0), statfun=numpy.amax))
#obs.append(LocalOpticsObservable([idx_sext, idx_center], phase_advance, summary=True, target=0.5, bounds=(0.0, 0.1)))

newtl2 = match(tl2, variables, obs, copy=True, max_nfev=100)
newtl2.plot_beta(twiss_in=twiin)

_, _, ld = newtl2.get_optics(refpts=at.All, twiss_in=twiin, method=at.linopt2)
idx_sext = tl2.get_uint32_index('SEXT')[0]
idx_center = tl2.get_uint32_index('Center')[0]
idx_nlks = tl2.get_uint32_index('NLK*')
print((ld.mu[idx_center]-ld.mu[idx_sext])/numpy.pi/2)
for i in idx_nlks:
    print(tl2[i].FamName, (ld.mu[i] - ld.mu[idx_sext]) / numpy.pi / 2)
#for i in range(len(tl2)):
#   print(tl2[i].FamName, (ld.mu[i] - ld.mu[idx_center]) / numpy.pi / 2)

#_, _, ld = at.get_optics(tl2, refpts=at.All, twiss_in=l0)
#plt.plot(spos, ld.beta[:, 0], label='betax')
#plt.plot(spos, ld.beta[:, 1], label='betay')
#plt.plot(spos, ld.dispersion[:, 0]*1.0e2, label='Dx')
#plt.legend()

# plt.savefig('/machfs/sauret/SharedCodes/tracking/data_sim/19_03_25/TL2_no_gradient.png', dpi=100)
plt.show()
# at.save_lattice(newtl2, '/machfs/sauret/SharedCodes/tracking/lattices/newtl2_7b_no_gradient.mat')