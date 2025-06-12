import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

path = '/machfs/sauret/SharedCodes/tracking/data_sim/18_03_25/'

data = pickle.load(open(path+'beta7_data_EBS.pkl', 'rb'))
data2 = pickle.load(open(path+'beta7_data_EBS_low_emit.pkl', 'rb'))
data3 = pickle.load(open(path+'beta7_data_EBS_full_energy.pkl', 'rb'))


x = data['x']
y = data["xp"]
ie = data["ie"]

x2 = data2['x']
y2 = data2["xp"]
ie2 = data2["ie"]

x3 = data3['x']
y3 = data3["xp"]
ie3 = data3["ie"]


fix, ax = plt.subplots()
ax.contourf(x3, y3, ie3, 50)
CS = ax.contour(x3, y3, ie3, [0.50,0.60,0.70,0.80,0.90], colors='black')
ax.clabel(CS, colors='black')
ax.set_xlabel(r"x [m]")
ax.set_ylabel(r"xp [rad]")
plt.tight_layout()
plt.savefig(path+'beta7_data_EBS_full_energy.png', dpi=100)
plt.show()

# plt.plot(x, ie[y==0].T*100, label='b = 7m')
# plt.plot(x2, ie2[y2==0].T*100, label='b = 7m - low emit')
plt.plot(x3, ie3[y3==0].T*100, label='b = 7m - full energy')
plt.xlabel(r'x [m]')
plt.ylabel(r'I.E. [%]')
plt.legend()
plt.tight_layout()
plt.savefig(path+'beta7_data_EBS_full_energy_cut.png',dpi=100)
plt.show()
