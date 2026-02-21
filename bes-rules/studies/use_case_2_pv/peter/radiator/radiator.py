import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

ren = {
    "radCon.Q_flow": "Qdot",
    "sou.T_in": "T_forward",
    #"radCon.vol[1].T": "",
    "radCon.vol[5].T": "T_return",
}
df = pd.read_csv("n_1.01_sim_param.csv").rename(mapper=ren, axis="columns")

c = 4184
Q_flow_nominal = 5527.774735185678  # 5000
T_a_nominal = 328.15
T_b_nominal = 318.15
frad = 0.35
TAir_nominal = 293.15
TRad_nominal = TAir_nominal
mTra_flow_nominal = 0.1321169869786252   # Q_flow_nominal/(T_a_nominal - T_b_nominal)/ c
UA = Q_flow_nominal / ((T_a_nominal + T_b_nominal) / 2 - ((1 - frad) * TAir_nominal + frad * TRad_nominal))

valve = 1
mdot = mTra_flow_nominal * valve
T_forward = df["T_forward"]
T_Air = TAir_nominal

#T_return = T_Air + (T_forward - T_Air) * np.exp(-UA / (mdot * c))
#Qdot = mdot * c * (T_forward - T_return)

UA_Ele = UA / 5
TTraSup = T_forward
T_heater_1 = (mdot * c * TTraSup + UA_Ele * T_Air) / (mdot * c + UA_Ele)
T_heater_2 = (mdot * c * T_heater_1 + UA_Ele * T_Air) / (mdot * c + UA_Ele)
T_heater_3 = (mdot * c * T_heater_2 + UA_Ele * T_Air) / (mdot * c + UA_Ele)
T_heater_4 = (mdot * c * T_heater_3 + UA_Ele * T_Air) / (mdot * c + UA_Ele)
T_heater_5 = (mdot * c * T_heater_4 + UA_Ele * T_Air) / (mdot * c + UA_Ele)
T_return = T_heater_5
Q_heater_1 = UA_Ele * (T_heater_1 - T_Air)
Q_heater_2 = UA_Ele * (T_heater_2 - T_Air)
Q_heater_3 = UA_Ele * (T_heater_3 - T_Air)
Q_heater_4 = UA_Ele * (T_heater_4 - T_Air)
Q_heater_5 = UA_Ele * (T_heater_5 - T_Air)
Qdot = Q_heater_1 + Q_heater_2 + Q_heater_3 + Q_heater_4 + Q_heater_5

# plotting
fig, ax = plt.subplots(nrows=2, ncols=3, sharex="col", figsize=(15, 10), layout="constrained")

ax[0][0].plot(T_forward, Qdot)
ax[0][0].plot(df["T_forward"], -df["Qdot"])
ax[0][0].set_ylabel("Qdot")
ax[0][1].plot(T_forward, Qdot/-df["Qdot"])
ax[0][1].set_ylabel("Quotient Qdot")
ax[0][2].plot(T_forward, Qdot--df["Qdot"])
ax[0][2].set_ylabel("Differenz Qdot")

ax[1][0].plot(T_forward, T_return)
ax[1][0].plot(df["T_forward"], df["T_return"])
ax[1][0].set_ylabel("T_return")
ax[1][1].plot(T_forward, T_return/df["T_return"])
ax[1][1].set_ylabel("Quotient T_return")
ax[1][2].plot(T_forward, T_return-df["T_return"])
ax[1][2].set_ylabel("Differenz T_return")

ax[1][0].set_xlabel("T_forward")
ax[1][1].set_xlabel("T_forward")
ax[1][2].set_xlabel("T_forward")

for a in ax:
    for aa in a:
        aa.grid()

ax[0][0].legend(["MPC", "BESMod", "BESMod fit"])
ax[1][0].legend(["MPC", "BESMod"])

plt.savefig("radiator.png", format="png")
plt.savefig("radiator.svg", format="svg")

plt.show()




