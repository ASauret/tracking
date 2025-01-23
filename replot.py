import pickle
import matplotlib.pyplot as plt

data = pickle.load(open('data.pkl', 'rb'))

x = data['k1']
y = data["k2"]
ie = data["ie"]

print(ie)
print(x)
print(y)
exit()

fix, ax = plt.subplots()
ax.contourf(x, y, ie, 50)
CS = ax.contour(x, y, ie, levels=[0.9, 0.94, 0.98], colors='black')
ax.clabel(CS, colors='black')
ax.set_xlabel(r"x' [rad]")
ax.set_ylabel(r"k$_1$L [m$^{-1}$]")
plt.show()