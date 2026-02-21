# Read me

rbpc_optimization.py contains the generation of the files for the mpc simulation as well as the simulation of the rbpc.
pv.hdf contains the data of the photovoltaic system and THeaCur.csv the data of the heatcurve, which are needed for the disturbance prediction.

MPC_simulation contains the agents, parameter and results of the MPC simulation.

rbpc_optimization.py contains the development of the rbpc. It needs the results of the MPC simulation.

save_path.txt contains the path where the current results of the rbpc are saved. Its a dirty workaround for the plotting of the simulation results with RBPC/plot_rbpc_sim.py

evaluation contains the calculation and plotting tools for the comparison of the controls.
The timeseries lay in MPC/, RBC/, RBC_noSGReady/, RBPC/ and RBPC_ZCBE/-folders.
The results of the evaluation is in the respective evaluation/results/-folders.
