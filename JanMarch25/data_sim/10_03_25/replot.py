import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

path = '/machfs/sauret/SharedCodes/tracking/data_sim/10_03_25/'

data = pickle.load(open(path+'data_EBS_sextu130.pkl', 'rb'))
data2 = pickle.load(open(path+'data_EBS_sextu130_moins.pkl', 'rb'))

x = data['x']
y = data["xp"]
ie = data["ie"]

x2 = data2['x']
y2 = data2["xp"]
ie2 = data2["ie"]

fix, ax = plt.subplots()
ax.contourf(x2, y2, ie2, 50)
CS = ax.contour(x2, y2, ie2, [0.50,0.60,0.70,0.80,0.90], colors='black')
ax.clabel(CS, colors='black')
ax.set_xlabel(r"x [m]")
ax.set_ylabel(r"xp [rad]")
plt.tight_layout()
plt.savefig(path+'data_EBS_sextu120.png', dpi=100)
plt.show()

# plt.plot(x, ie[y==0].T, label='b = 5m, sextupole compensation 130 T/m2')
plt.plot(x2, ie2[y2==0].T, label='b = 5m, sextupole compensation -130 T/m2')
plt.xlabel(r'x [m]')
plt.ylabel(r'I.E. [%]')
plt.legend()
plt.tight_layout()
plt.savefig(path+'data_EBS_sextu120_cut.png',dpi=100)
plt.show()
