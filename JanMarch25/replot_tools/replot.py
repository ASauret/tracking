import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

path = '/machfs/sauret/SharedCodes/tracking/data_sim/divers/'

data7b = pickle.load(open(path+'data_EBS_7b.pkl', 'rb'))
data5b = pickle.load(open(path+'data_EBS_5b.pkl', 'rb'))
data10b = pickle.load(open(path+'data_EBS_10b.pkl', 'rb'))
data2 = pickle.load(open(path+'data_EBS2.pkl', 'rb'))
data_le = pickle.load(open(path+'data_EBS_low_emit.pkl', 'rb'))
data_le30 = pickle.load(open(path+'data_EBS_low_emit30_p.pkl', 'rb'))
data_7_8 = pickle.load(open(path+'data_EBS_7_8.pkl', 'rb'))
data_thin = pickle.load(open(path+'data_EBS_thinNLK_5b_loe_8.pkl', 'rb'))
data_sextu = pickle.load(open(path+'data_EBS_no_qf1sr.pkl', 'rb'))
data_nlk = pickle.load(open(path+'booster_upgrade/data_EBS_2d_runners.pkl', 'rb'))
data_1d = pickle.load(open(path+'booster_upgrade/data_EBS_1d_runners.pkl', 'rb'))

x = data7b['x']
y = data7b["xp"]
ie = data7b["ie"]

x5 = data5b['x']
y5 = data5b["xp"]
ie5 = data5b["ie"]

x10 = data10b['x']
y10 = data10b["xp"]
ie10 = data10b["ie"]

x2 = data2['x']
y2 = data2["xp"]
ie2 = data2["ie"]

xle = data_le['x']
yle = data_le["xp"]
iele = data_le["ie"]

xle30 = data_le30['x']
yle30 = data_le30["xp"]
iele30 = data_le30["ie"]

x78 = data_7_8['x']
y78 = data_7_8["xp"]
ie78 = data_7_8["ie"]

x_thin = data_thin['x']
y_thin = data_thin["xp"]
ie_thin = data_thin["ie"]

x_sextu = data_sextu['x']
y_sextu = data_sextu["xp"]
ie_sextu = data_sextu["ie"]

x_nlk = data_nlk['x']
y_nlk = data_nlk["xp"]
ie_nlk = data_nlk["ie"]

x_1d = data_1d['x']
ie_1d = data_1d["ie"]

fix, ax = plt.subplots()
ax.contourf(x_nlk, y_nlk, ie_nlk, 50)
CS = ax.contour(x_nlk, y_nlk, ie_nlk, [0.50,0.60,0.70,0.80,0.90], colors='black')
ax.clabel(CS, colors='black')
ax.set_xlabel(r"x [m]")
ax.set_ylabel(r"xp [rad]")
plt.tight_layout()
plt.show()

ie_thin_nlk = np.loadtxt('/machfs/sauret/SharedCodes/tracking/ie.txt')
offset_scan = np.linspace(-0.003, 0.003, 31)

plt.plot(x, ie[y==0].T, label='b = 7m')
plt.plot(x5, ie5[y5==0].T, label='b = 5m')
plt.plot(x10, ie10[y10==0].T,label='b = 10m')
plt.plot(x2, ie2[y2==0].T,label='maille qd3')
plt.plot(xle, iele[yle==0].T,label='low emit')
plt.plot(xle30, iele30[yle30==0].T,label='low emit 30%')
# plt.plot(x78, ie78[:,y78==0],label='7.8mm ')
# plt.plot(x_nlk, ie_nlk[y_nlk==0.00].T,label='only NLK')
# plt.plot(x_sextu, ie_sextu[:,y_sextu==0],label='Compensation sextu')
# plt.plot(offset_scan, ie_thin_nlk/100, label='thin raw')
# plt.plot(x_1d, ie_1d, label='1D runner')
plt.xlabel(r'x [m]')
plt.ylabel(r'I.E. [%]')
plt.legend()
plt.show()
