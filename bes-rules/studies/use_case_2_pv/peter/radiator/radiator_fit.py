import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def Qdot_fit2(T_forward, a, b, c):
    return a + b * T_forward + c * T_forward ** 2


def Qdot_fit3(T_forward, a, b, c, d):
    return a + b * T_forward + c * T_forward ** 2 + d * T_forward ** 3


ren = {
    "radCon.Q_flow": "Qdot",
    "sou.T_in": "T_forward",
    #"radCon.vol[1].T": "",
    "radCon.vol[5].T": "T_return",
}
df = pd.read_csv("n_1.24_sim_param.csv").rename(mapper=ren, axis="columns")

xdata = df["T_forward"]
ydata = df["Qdot"]
popt2, pcov2 = sp.optimize.curve_fit(Qdot_fit2, xdata, ydata)
popt3, pcov3 = sp.optimize.curve_fit(Qdot_fit3, xdata, ydata)
Q_fit2 = -Qdot_fit2(df["T_forward"], a=popt2[0], b=popt2[1], c=popt2[2])
Q_fit3 = -Qdot_fit3(df["T_forward"], a=popt3[0], b=popt3[1], c=popt3[2], d=popt3[3])

# plotting
fig, ax = plt.subplots(nrows=1, ncols=3, sharex="col", figsize=(15, 5), layout="constrained")

ax[0].plot(df["T_forward"], Q_fit2)
ax[0].plot(df["T_forward"], Q_fit3)
ax[0].plot(df["T_forward"], -df["Qdot"])
ax[0].set_ylabel("Qdot")
ax[1].plot(df["T_forward"], Q_fit2/-df["Qdot"])
ax[1].plot(df["T_forward"], Q_fit3/-df["Qdot"])
ax[1].set_ylabel("Quotient Qdot")
ax[2].plot(df["T_forward"], Q_fit2--df["Qdot"])
ax[2].plot(df["T_forward"], Q_fit3--df["Qdot"])
ax[2].set_ylabel("Differenz Qdot")

ax[0].set_xlabel("T_forward")
ax[1].set_xlabel("T_forward")
ax[2].set_xlabel("T_forward")

for a in ax:
    a.grid()

ax[0].legend(["BESMod Polynom n=2", "BESMod Polynom n=3", "BESMod"])
ax[1].legend(["BESMod Polynom n=2", "BESMod Polynom n=3"])
ax[2].legend(["BESMod Polynom n=2", "BESMod Polynom n=3"])

plt.savefig("radiator_fit.png", format="png")
plt.savefig("radiator_fit.svg", format="svg")

plt.show()



