import pickle
import matplotlib.pyplot as plt

data = pickle.load(open('/machfs/sauret/data_EBS1_debug.pkl', 'rb'))

x = data['x']
y = data["xp"]
ie = data["ie"]

# print(ie)
# exit()
# print(x.shape)
# print('/n')
# print(y.shape)
# exit()

fix, ax = plt.subplots()
ax.contourf(x, y, ie, 50)
CS = ax.contour(x, y, ie, colors='black')
ax.clabel(CS, colors='black')
ax.set_xlabel(r"x [m]")
ax.set_ylabel(r"xp [rad]")
plt.tight_layout()
plt.show()