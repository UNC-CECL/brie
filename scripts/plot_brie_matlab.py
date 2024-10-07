"""
Written by K.Anarde

- imports matlab inputs for seeding of brie_org.py (for version testing and grid testing)

"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from brie.brie import Brie

os.chdir(
    "/Users/rsahrae/PycharmProjects/pythonProject_CE_roya/scripts/"
)

# subset of indices for testing grid discretization
name = "test_brie_matlab_seed"
dt = [0.05, 0.1, 0.25, 0.50, 1]  # yr
dy = [1000, 500, 250, 100, 50]  # m

# import structures from Matlab
mat = loadmat("test_brie_matlab_seed.mat")

output = np.empty((np.size(dt), np.size(dy)), dtype=object)


def batchBrie(ii, jj, dt, dy, name, mat):
    # get matlab seed parameters
    xs = [mat["output"][ii][jj]["xs"].flat[0]]
    xs = np.r_[xs[0]].flatten()  # this is a numpy array - make sure that's ok
    wave_angle = [mat["output"][ii][jj]["wave_angle"].flat[0]]
    wave_angle = list(np.r_[wave_angle[0]].flatten())  # this is a list
    print("size of Matlab wave_angle and xs = ", np.size(wave_angle), np.size(xs))

    # create a Brie instance
    brie = Brie(
        name=name,
        bseed=True,
        wave_height=1.0,
        wave_period=7,
        barrier_width_critical=450.0,
        barrier_height_critical=1.9,
        alongshore_section_length=dy[jj],
        time_step=dt[ii],
        time_step_count=1e3 / dt[ii],
        wave_angle=wave_angle,
        xs=xs,
    )  # initialize class

    # check that my updated variables are correct
    print("dt = ", dt[ii])
    print("dy = ", dy[jj])
    print("model timesteps for 1000 morphologic years = ", int(brie._nt))

    # run the brie model
    for time_step in range(int(brie._nt) - 1):
        brie.update()  # update the model by a time step

    # finalize by deleting variables and make Qinlet m^3/yr
    # brie.finalize() LVB commented out this function

    return brie


###############################################################################
# run model
###############################################################################

inputs_dt = range(np.size(dt))
inputs_dy = range(np.size(dy))

for kk in inputs_dt:
    for mm in inputs_dy:
        model_out = batchBrie(kk, mm, dt, dy, name, mat)
        output[kk, mm] = model_out  # ([b_out, param1, param2, param])

###############################################################################
# save output
###############################################################################

filename = name + ".npz"
np.savez(filename, output=output)

###############################################################################
# plots
###############################################################################

# preallocate arrays
(
    Qoverwash_total_py,
    Qoverwash_total_mat,
    Qinlet_total_py,
    Qinlet_total_mat,
    F_py,
    F_mat,
    dy_py,
    dt_py,
    nt_py,
    dy_mat,
    dt_mat,
) = [np.zeros((np.size(dt), np.size(dy))) for _ in range(11)]

for ii in inputs_dt:
    for jj in inputs_dy:

        # matrix format is dt (row) x dy (column)
        brie = output[ii, jj]
        dy_py[ii, jj] = brie._dy  # for debugging
        dt_py[ii, jj] = brie._dt
        nt_py[ii, jj] = np.size(brie._Qoverwash)

        # find the mean of Qoverwash and Qinlet for calculation of F (see Jaap's Figure 9)
        Qoverwash_total_py[ii, jj] = np.mean(brie._Qoverwash)
        Qinlet_total_py[ii, jj] = np.mean(brie._Qinlet)
        F_py[ii, jj] = Qinlet_total_py[ii, jj] / (
            Qinlet_total_py[ii, jj] + Qoverwash_total_py[ii, jj]
        )

        inlet_age_tmp = brie._inlets._inlet_age.copy()
        for i in range(len(inlet_age_tmp)):
            inlet_age_tmp[i] = [inlet_age_tmp[i][0],inlet_age_tmp[i][1][0]]


        inlet_age_py = np.array(
            inlet_age_tmp
        )  # we are only going to look at the last model output

        # and now matlab output
        tmp_Qoverwash_mat = [mat["output"][ii][jj]["Qoverwash"].flat[0]]
        tmp_Qinlet_mat = [mat["output"][ii][jj]["Qinlet"].flat[0]]

        tmp_Qoverwash_mat = np.r_[tmp_Qoverwash_mat[0]].flatten()
        tmp_Qinlet_mat = np.r_[tmp_Qinlet_mat[0]].flatten()
        tmp_dy_mat = [mat["output"][ii][jj]["dy"].flat[0]]
        dy_mat[ii, jj] = np.r_[tmp_dy_mat[0]].flatten()
        tmp_dt_mat = [mat["output"][ii][jj]["dt"].flat[0]]
        dt_mat[ii, jj] = np.r_[tmp_dt_mat[0]].flatten()
        tmp_inlet_age_mat = [mat["output"][ii][jj]["inlet_age"].flat[0]]
        inlet_age_mat = tmp_inlet_age_mat[0]

        # again, find the mean
        Qoverwash_total_mat[ii, jj] = np.mean(tmp_Qoverwash_mat)
        Qinlet_total_mat[ii, jj] = np.mean(tmp_Qinlet_mat)
        F_mat[ii, jj] = Qinlet_total_mat[ii, jj] / (
            Qinlet_total_mat[ii, jj] + Qoverwash_total_mat[ii, jj]
        )

# plot the inlet age for the last model
fig, axs = plt.subplots(1, 2)

ax = axs[0]
ax.scatter(inlet_age_py[:, 0], inlet_age_py[:, 1])
ax.set_title("python - inlet_age")
ax.set_xlabel("timestep (dt)")
ax.set_ylabel("inlet id")

ax = axs[1]
ax.scatter(inlet_age_mat[:, 0], inlet_age_mat[:, 1])
ax.set_xlabel("dt (yr)")
ax.set_xlabel("timestep (dt)")
ax.set_title("matlab - inlet_age")
plt.show()

# histogram of F
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].hist(F_py)
axs[0].set_xlabel("F")
axs[0].set_ylabel("Number of model runs")
axs[0].set_title("python")

axs[1].hist(F_mat)
axs[1].set_xlabel("F")
axs[1].set_ylabel("Number of model runs")
axs[1].set_title("matlab")

# grid discretization comparison (no normalization, no transpose)
fig, axs = plt.subplots(2, 3, sharey=True)
plt.jet()

# Qoverwash
ax = axs[0, 1]
im = ax.pcolormesh(
    Qoverwash_total_py, edgecolors="white", linewidths=1, antialiased=True
)  # , vmin=0.8,vmax=1.2
fig.colorbar(im, ax=ax)
ax.set_title("python - Qoverwash")
ax.set_yticks(np.arange(len(dt)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(dy)) + 0.5)
ax.set_xticklabels(dy)
ax.set_yticklabels(dt)  # dt

ax = axs[1, 1]
im = ax.pcolormesh(
    Qoverwash_total_mat, edgecolors="white", linewidths=1, antialiased=True
)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - Qoverwash")
ax.set_yticks(np.arange(len(dt)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(dy)) + 0.5)
ax.set_xticklabels(dy)
ax.set_yticklabels(dt)  # dt
ax.set_xlabel("dy (m)")

# Qinlet
ax = axs[0, 0]
im = ax.pcolormesh(Qinlet_total_py, edgecolors="white", linewidths=1, antialiased=True)
fig.colorbar(im, ax=ax)
ax.set_title("python - Qinlet")
ax.set_yticks(np.arange(len(dt)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(dy)) + 0.5)
ax.set_xticklabels(dy)
ax.set_yticklabels(dt)  # dt
ax.set_ylabel("dt (yr)")

ax = axs[1, 0]
im = ax.pcolormesh(Qinlet_total_mat, edgecolors="white", linewidths=1, antialiased=True)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - Qinlet")
ax.set_yticks(np.arange(len(dt)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(dy)) + 0.5)
ax.set_xticklabels(dy)
ax.set_yticklabels(dt)  # dt
ax.set_xlabel("dy (m)")
ax.set_ylabel("dt (yr)")

# F
ax = axs[0, 2]
im = ax.pcolormesh(F_py, edgecolors="white", linewidths=1, antialiased=True)
fig.colorbar(im, ax=ax)
ax.set_title("python - F")
ax.set_yticks(np.arange(len(dt)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(dy)) + 0.5)
ax.set_xticklabels(dy)
ax.set_yticklabels(dt)  # dt

ax = axs[1, 2]
im = ax.pcolormesh(F_mat, edgecolors="white", linewidths=1, antialiased=True)
fig.colorbar(im, ax=ax)
ax.set_title("matlab - F")
ax.set_yticks(np.arange(len(dt)) + 0.5)  # set ticks in the center of box
ax.set_xticks(np.arange(len(dy)) + 0.5)
ax.set_xticklabels(dy)
ax.set_yticklabels(dt)  # dt
ax.set_xlabel("dy (m)")

# debugging the new inlet modifications from Jaap
# I think the differences in the indices inlet_idx (for the same shoreline and wave angle) between the matlab and python
# models come from rounding errors on x_s when imported into python (not enough precision), but this should be examined
# further as a rewrite the inlet model for Barrier3D (i.e., the inlet dynamics appear to be working properly (see
# comparison of inlet_age in plot_brie_colorgrid.py)

# from scipy.io import loadmat
# from brie import Brie
# import numpy as np
# import matplotlib.pyplot as plt
#
# mat = loadmat('test.mat')
# xs = [mat['x_s']]
# xs = np.r_[xs[0]].flatten()
# wave_angle = [mat['wave_ang_save']]
# wave_angle = list(np.r_[wave_angle[0]].flatten())
#
# model = Brie()
#
# model._bseed = True
# model._h_b_crit = 1.9
# model._w_b_crit = 450
# model._wave_height = 1
# model._wave_period = 7
#
# Brie.dependent(model, wave_angle=wave_angle, xs=xs)
#
# for time_step in range(int(model._nt) - 1):
#     Brie.update(model)
#
# plt.plot(model._Qinlet * model._dt / (model._dy * model._ny) ) # Qinlet is in m3/yr --> m3/m/yr
# plt.plot(model._Qoverwash / (model._dy * model._ny) )
