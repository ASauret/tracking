import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

path = '/machfs/sauret/SharedCodes/tracking/data_sim/05_03_25/'

data = pickle.load(open(path+'data_EBS.pkl', 'rb'))
data2 = pickle.load(open(path+'data_EBS_minus8_5.pkl', 'rb'))
data3 = pickle.load(open(path+'data_EBS_minus8_5_low_emit.pkl', 'rb'))
data4 = pickle.load(open(path+'data_EBS_minus8_5_linac6gev.pkl','rb'))

x = data['x']
y = data["xp"]
ie = data["ie"]

x2 = data2['x']
y2 = data2["xp"]
ie2 = data2["ie"]

x3 = data3['x']
y3 = data3["xp"]
ie3 = data3["ie"]

x4 = data4['x']
y4 = data4["xp"]
ie4 = data4["ie"]

fix, ax = plt.subplots()
ax.contourf(x3, y3, ie3, 50)
CS = ax.contour(x3, y3, ie3, [0.50,0.60,0.70,0.80,0.90], colors='black')
ax.clabel(CS, colors='black')
ax.set_xlabel(r"x [m]")
ax.set_ylabel(r"xp [rad]")
plt.tight_layout()
plt.savefig('/machfs/sauret/SharedCodes/tracking/data_sim/05_03_25/data_EBS_minus8_5_low_emit.png', dpi=100)
plt.show()

# plt.plot(x, ie[y==0].T, label='b = 6.05m')
# plt.plot(x2, ie2[y2==0].T, label='b = 6.05m, -8.5mm')
plt.plot(x3, ie3[y3==0].T, label='b = 6.05m, -8.5mm, low emit')
# plt.plot(x4, ie4[y4==0].T, label='b = 6.05m, -8.5mm, full energy linac')
plt.xlabel(r'x [m]')
plt.ylabel(r'I.E. [%]')
plt.legend()
plt.savefig('/machfs/sauret/SharedCodes/tracking/data_sim/05_03_25/data_EBS minus8_5_low_emit_cut.png', dpi=100)
plt.show()
