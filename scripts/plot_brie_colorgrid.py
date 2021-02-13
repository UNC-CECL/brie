#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:53:45 2020

@author: KatherineAnardeWheels
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

#%%
###############################################################################
# load stuff
###############################################################################

# python outputs
output = np.load("CASCADE_comparison.npz", allow_pickle=True)
output = output["output"]

# matlab structures
mat = loadmat("CASCADE_comparison.mat")

#%%
###############################################################################
# iterate through both outputs and get overwash and inlet flux
###############################################################################

# param1 = [0.01, 0.02, 0.025, 0.05, 0.08, 0.1, 0.2, 0.25]
# param2 = [1000, 800, 500, 400, 250, 100, 80, 50]
param1 = [0.05, 0.1, 0.25, 0.50, 1]  # yr
param2 = [1000, 500, 250, 100, 50]  # , 10]   # m

inputs_p1 = range(np.size(param1))
inputs_p2 = range(np.size(param2))

# preallocate arrays
(
    Qoverwash_total,
    Qoverwash_mat_total,
    Qinlet_total,
    Qinlet_mat_total,
    F,
    F_mat,
    dy,
    dt,
    nt,
    dy_mat,
    dt_mat,
) = [np.zeros((np.size(param1), np.size(param2))) for _ in range(11)]

# normalize transgressive sediment flux?
bnorm = False  # buggy

for ii in inputs_p1:
    for jj in inputs_p2:

        # matrix format is dt (row) x dy (column)
        model_out = output[ii, jj]
        dy[ii, jj] = model_out._dy  # for debugging
        dt[ii, jj] = model_out._dt
        nt[ii, jj] = np.size(model_out._Qoverwash)

        # sum outputs for the same morphologic time (here, calculated as 1,000 years)
        if bnorm:
            Qoverwash_total[ii, jj] = np.mean(
                model_out._Qoverwash / (model_out._ny * model_out._dy)
            )
            Qinlet_total[ii, jj] = np.mean(
                model_out._Qinlet / (model_out._ny * model_out._dy)
            )
        else:
            Qoverwash_total[ii, jj] = np.mean(model_out._Qoverwash)
            Qinlet_total[ii, jj] = np.mean(model_out._Qinlet)

        F[ii, jj] = Qinlet_total[ii, jj] / (
            Qinlet_total[ii, jj] + Qoverwash_total[ii, jj]
        )
        # inlet_age = [model_out._inlet_age]
        inlet_nr = [model_out._inlet_nr]

        # get matlab seed parameters
        if bnorm:
            tmp_Qoverwash_mat = [mat["output"][ii][jj]["Qoverwash"].flat[0]]
            tmp_Qinlet_mat = [mat["output"][ii][jj]["Qinlet"].flat[0]]
            tmp_Qoverwash_mat = tmp_Qoverwash_mat / (model_out._ny * model_out._dy)
            tmp_Qinlet_mat = tmp_Qinlet_mat / (model_out._ny * model_out._dy)
        else:
            tmp_Qoverwash_mat = [mat["output"][ii][jj]["Qoverwash"].flat[0]]
            tmp_Qinlet_mat = [mat["output"][ii][jj]["Qinlet"].flat[0]]

        tmp_Qoverwash_mat = np.r_[tmp_Qoverwash_mat[0]].flatten()
        tmp_Qinlet_mat = np.r_[tmp_Qinlet_mat[0]].flatten()
        tmp_dy_mat = [mat["output"][ii][jj]["dy"].flat[0]]
        dy_mat[ii, jj] = np.r_[tmp_dy_mat[0]].flatten()
        tmp_dt_mat = [mat["output"][ii][jj]["dt"].flat[0]]
        dt_mat[ii, jj] = np.r_[tmp_dt_mat[0]].flatten()
        tmp_inlet_nr_mat = [mat["output"][ii][jj]["inlet_nr"].flat[0]]
        inlet_nr_mat = np.r_[tmp_inlet_nr_mat[0]].flatten()
        # tmp_inlet_age_mat = [mat['output'][ii][jj]['inlet_age'].flat[0]]
        # inlet_age_mat[ii,jj] = np.r_[tmp_inlet_age_mat[0]].flatten()

        Qoverwash_mat_total[ii, jj] = np.mean(tmp_Qoverwash_mat[0 : int(nt[ii, jj])])
        Qinlet_mat_total[ii, jj] = np.mean(tmp_Qinlet_mat[0 : int(nt[ii, jj])])

        F_mat[ii, jj] = Qinlet_mat_total[ii, jj] / (
            Qinlet_mat_total[ii, jj] + Qoverwash_mat_total[ii, jj]
        )

        # INLET AGE and # of inlets active through time
#        fig, axs = plt.subplots(2, 2)
#
#        t = np.arange(dt[ii,jj],(dt[ii,jj]*nt[ii,jj])+dt[ii,jj],model_out._dtsave*dt[ii,jj])  # time array
#        ax = axs[0,0]
#        ax.scatter(t, inlet_nr)
#        ax.set_title('python - inlet_nr')
#        ax.set_xlabel('dt (yr)')
#        ax.set_ylabel('# inlets active')
#
#        ax = axs[0,1]
#        ax.scatter(t, inlet_nr_mat)
#        ax.set_title('matlab - inlet_nr')
#        ax.set_xlabel('dt (yr)')
#        ax.set_ylabel('# inlets active')
#        plt.show()


# normalize fluxes by dy to make m3/m/yr
# Qoverwash_total = Qoverwash_total/dy
# Qoverwash_mat_total = Qoverwash_mat_total/dy
# Qinlet_total = Qinlet_total/dy
# Qinlet_mat_total = Qinlet_mat_total/dy
#
#%%
###############################################################################
# plot - original testing of transpose and normalization
###############################################################################

fig, axs = plt.subplots(2, 2)
plt.jet()

# Qoverwash
ax = axs[0, 0]
# in the current setup, [3,5 corresponds to dt = 0.05, dy = 100]
# CACSCADE setup: [0,3] corresponds to 0.05, 100
# [5,3] corresponds to dt = 0.1, dy = 400]
im = ax.pcolormesh(
    np.transpose(Qoverwash_total / Qoverwash_total[0, 3]),
    edgecolors="white",
    linewidths=1,
    antialiased=True,
)  # , vmin=0.8,vmax=1.2
fig.colorbar(im, ax=ax)
ax.set_title("python - Qoverwash")
ax.set_xticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param1)  # dt
# ax.set_yticklabels(reversed(param2))
ax.set_yticklabels(param2)
ax.set_xlabel("dt (yr)")
ax.set_ylabel("dy (m)")

ax = axs[0, 1]
im = ax.pcolormesh(
    np.transpose(Qoverwash_mat_total / Qoverwash_mat_total[0, 3]),
    edgecolors="white",
    linewidths=1,
    antialiased=True,
)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - Qoverwash")
ax.set_xticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param1)  # dt
ax.set_yticklabels(param2)
ax.set_xlabel("dt (yr)")
ax.set_ylabel("dy (m)")

# Qinlet
ax = axs[1, 0]
im = ax.pcolormesh(
    np.transpose(Qinlet_total / Qinlet_total[0, 3]),
    edgecolors="white",
    linewidths=1,
    antialiased=True,
)
fig.colorbar(im, ax=ax)
ax.set_title("python - Qinlet")
ax.set_xticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param1)  # dt
ax.set_yticklabels(param2)
ax.set_xlabel("dt (yr)")
ax.set_ylabel("dy (m)")

ax = axs[1, 1]
im = ax.pcolormesh(
    np.transpose(Qinlet_mat_total / Qinlet_mat_total[0, 3]),
    edgecolors="white",
    linewidths=1,
    antialiased=True,
)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - Qinlet")
ax.set_xticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param1)  # dt
ax.set_yticklabels(param2)
ax.set_xlabel("dt (yr)")
ax.set_ylabel("dy (m)")


# the following code is for debugging the dy dt plotting scheme in python
# ax = axs[2,0]
# im = ax.pcolormesh(np.transpose(dy), edgecolors='white', linewidths=1,
#                   antialiased=True)
# fig.colorbar(im, ax=ax)
# ax.set_title('python - dy')
#
# ax = axs[3,0]
# im = ax.pcolormesh(np.transpose(dt), edgecolors='white', linewidths=1,
#                   antialiased=True)
# fig.colorbar(im, ax=ax)
# ax.set_title('python - dt')
#
# ax = axs[2,1]
# im = ax.pcolormesh(np.transpose(dy_mat), edgecolors='white', linewidths=1,
#                   antialiased=True)
# fig.colorbar(im, ax=ax)
# ax.set_title('matlab - dy')
#
# ax = axs[3,1]
# im = ax.pcolormesh(np.transpose(dt_mat), edgecolors='white', linewidths=1,
#                   antialiased=True)
# fig.colorbar(im, ax=ax)
# ax.set_title('matlab - dt')

fig.tight_layout()
plt.show()

#%%
###############################################################################
# plot - no transpose
###############################################################################

fig, axs = plt.subplots(2, 3, sharey=True)
plt.jet()

# Qoverwash
ax = axs[0, 1]
# in the current setup, [3,5 corresponds to dt = 0.05, dy = 100]
# [5,3] corresponds to dt = 0.1, dy = 400]
im = ax.pcolormesh(
    Qoverwash_total / Qoverwash_total[5, 3],
    edgecolors="white",
    linewidths=1,
    antialiased=True,
)  # , vmin=0.8,vmax=1.2
fig.colorbar(im, ax=ax)
ax.set_title("python - Qoverwash")
ax.set_xticks(np.arange(len(param2)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt

ax = axs[1, 1]
im = ax.pcolormesh(
    Qoverwash_mat_total / Qoverwash_mat_total[5, 3],
    edgecolors="white",
    linewidths=1,
    antialiased=True,
)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - Qoverwash")
ax.set_xticks(np.arange(len(param2)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt
ax.set_xlabel("dy (m)")

# Qinlet
ax = axs[0, 0]
im = ax.pcolormesh(
    Qinlet_total / Qinlet_total[5, 3],
    edgecolors="white",
    linewidths=1,
    antialiased=True,
)
fig.colorbar(im, ax=ax)
ax.set_title("python - Qinlet")
ax.set_xticks(np.arange(len(param2)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt
ax.set_ylabel("dt (yr)")

ax = axs[1, 0]
im = ax.pcolormesh(
    Qinlet_mat_total / Qinlet_mat_total[5, 3],
    edgecolors="white",
    linewidths=1,
    antialiased=True,
)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - Qinlet")
ax.set_xticks(np.arange(len(param2)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt
ax.set_xlabel("dy (m)")
ax.set_ylabel("dt (yr)")

# F
ax = axs[0, 2]
im = ax.pcolormesh(F / F[5, 3], edgecolors="white", linewidths=1, antialiased=True)
fig.colorbar(im, ax=ax)
ax.set_title("python - F")
ax.set_xticks(np.arange(len(param2)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt

ax = axs[1, 2]
im = ax.pcolormesh(
    F_mat / F_mat[5, 3], edgecolors="white", linewidths=1, antialiased=True
)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - F")
ax.set_xticks(np.arange(len(param2)) + 0.5)  # set ticks in the center of box
ax.set_yticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt
ax.set_xlabel("dy (m)")

#%%
###############################################################################
# plot - no normalization, no transpose
###############################################################################

fig, axs = plt.subplots(2, 3, sharey=True)
plt.jet()

# Qoverwash
ax = axs[0, 1]
im = ax.pcolormesh(
    Qoverwash_total, edgecolors="white", linewidths=1, antialiased=True
)  # , vmin=0.8,vmax=1.2
fig.colorbar(im, ax=ax)
ax.set_title("python - Qoverwash")
ax.set_yticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt

ax = axs[1, 1]
im = ax.pcolormesh(
    Qoverwash_mat_total, edgecolors="white", linewidths=1, antialiased=True
)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - Qoverwash")
ax.set_yticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt
ax.set_xlabel("dy (m)")

# Qinlet
ax = axs[0, 0]
im = ax.pcolormesh(Qinlet_total, edgecolors="white", linewidths=1, antialiased=True)
fig.colorbar(im, ax=ax)
ax.set_title("python - Qinlet")
ax.set_yticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt
ax.set_ylabel("dt (yr)")

ax = axs[1, 0]
im = ax.pcolormesh(Qinlet_mat_total, edgecolors="white", linewidths=1, antialiased=True)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - Qinlet")
ax.set_yticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt
ax.set_xlabel("dy (m)")
ax.set_ylabel("dt (yr)")

# F
ax = axs[0, 2]
im = ax.pcolormesh(F, edgecolors="white", linewidths=1, antialiased=True)
fig.colorbar(im, ax=ax)
ax.set_title("python - F")
ax.set_yticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt

ax = axs[1, 2]
im = ax.pcolormesh(F_mat, edgecolors="white", linewidths=1, antialiased=True)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - F")
ax.set_yticks(np.arange(len(param1)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(param2)) + 0.5)
ax.set_xticklabels(param2)
ax.set_yticklabels(param1)  # dt
ax.set_xlabel("dy (m)")


# ax = axs[1,1]
# im = ax.pcolormesh(dy_mat, edgecolors='white', linewidths=1,
#                   antialiased=True)
# fig.colorbar(im, ax=ax)
# ax.set_title('matlab - dt')

# fig.tight_layout()
# plt.show()

#%%
###############################################################################
# plot - historgram of F
###############################################################################


fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].hist(F)
axs[0].set_xlabel("F")
axs[0].set_ylabel("Number of model runs")
axs[0].set_title("python")

axs[1].hist(F_mat)
axs[1].set_xlabel("F")
axs[1].set_ylabel("Number of model runs")
axs[1].set_title("matlab")
