import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

path = '/machfs/sauret/injection/results/full_linac/'

data = pickle.load(open(path+'hor_ps_scan.pkl', 'rb'))

x = data['x']
xp = data["xp"]
ie = data["ie"]

fix, ax = plt.subplots()
ax.contourf(x*1e3, xp*1e3, ie, 50)
CS = ax.contour(x*1e3, xp*1e3, ie, [0.50,0.60,0.70,0.80,0.90], colors='black')
ax.clabel(CS, colors='black')
ax.set_xlabel(r"x [mm]")
ax.set_ylabel(r"xp [mrad]")
plt.tight_layout()
plt.show()

plt.plot(x*1e3, ie[xp==0].T*100, label='beta=5m')
plt.xlabel(r'x [mm]')
plt.ylabel(r'I.E. [%]')
plt.legend()
plt.show()
