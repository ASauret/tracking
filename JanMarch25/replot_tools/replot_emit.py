import pickle
import matplotlib.pyplot as plt
import numpy as np

path = '/machfs/sauret/SharedCodes/tracking/data_sim/divers/'

data_thin = pickle.load(open(path+'data_EBS_emit.pkl', 'rb'))

ex = data_thin['ex']
ey = data_thin['ey']
ie = data_thin['ie']

fix, ax = plt.subplots()
ax.contourf(ex, ey, ie, 50)
CS = ax.contour(ex, ey, ie, [0.50,0.60,0.70,0.80,0.90], colors='black')
ax.clabel(CS, colors='black')
ax.set_xlabel(r"ex [nm.rad]")
ax.set_ylabel(r"ey [nm.rad]")
plt.tight_layout()
plt.show()

# plt.plot(x, ie[:,y==0], label='b = 7m')
