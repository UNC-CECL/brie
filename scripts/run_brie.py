#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:21:49 2020

@author: KatherineAnardeWheels

- imports matlab inputs for seeding of brie.py (grid testing, CSDMS platform)

"""
import numpy as np

from scipy.io import loadmat
from brie import Brie
    
#%%
###############################################################################
# guts
###############################################################################

# subset of indices for testing grid discretization
name   = 'CASCADE_comparison'
param  = ['_dt','_dy'] 
#param1 = [0.01, 0.02, 0.025, 0.05, 0.08, 0.1, 0.2, 0.25]
#param2 = [1000, 800, 500, 400, 250, 100, 80, 50]
param1 = [0.05, 0.1, 0.25, 0.50, 1]  # yr
param2 = [1000, 500, 250, 100, 50]#, 10]   # m

inputs_p1 = range(np.size(param1))
inputs_p2 = range(np.size(param2))

# import structures from Matlab
mat = loadmat('CASCADE_comparison.mat')
#mat = loadmat('CASCADE_comparison_inlets_only.mat')

output = np.empty( (np.size(param1), np.size(param2)), dtype=object)
    
def batchBrie(ii, jj, param1, param2, param, name, mat):
    
    # get matlab seed parameters
    xs = [mat['output'][ii][jj]['xs'].flat[0]]
    xs = np.r_[xs[0]].flatten()    # this is a numpy array - make sure that's ok
    wave_angle = [mat['output'][ii][jj]['wave_angle'].flat[0]]
    wave_angle = list(np.r_[wave_angle[0]].flatten())  # this is a list
    print('size of Matlab wave_angle and xs = ', np.size(wave_angle), np.size(xs))
    
    # create an instance of the pymt Brie model
    model = Brie()  # this initializes the model and calls __init__
    
    # update the initial conditions
    model._name = name
    model._plot_on = False
    model._make_gif = False
    model._inlet_model_on = True
    model._barrier_model_on = True
    model._bseed = True    
    setattr(model, param[0], param1[ii]) # dt
    setattr(model, param[1], param2[jj]) # dy

    model._nt = 1e3/param1[ii]  # timesteps for 1000 morphologic years
    
    # v5 - test against Jaap's paper Figure 9 initial parameters
    #model._a0  = 1.5    # higher tidal amplitude (0.5 m)
    #model._slr = 3e-3   # higher slr (2e-3 m/yr)
    #model._wave_height = 1.5  # higher wave height (1 m)
    #model._wave_period = 8    # lower wave period (8 s)

    # CASCADE - test against parameters that make sense for LTA comparison
    model._h_b_crit = 1.9
    model._w_b_crit = 450
    model._wave_height = 1
    model._wave_period = 7

    # get dependent variables and seed wave angle and shoreline position
    Brie.dependent(model, wave_angle=wave_angle, xs=xs)
    
    # check that my updated variables are correct
    print('iterate through param1 = ', ii)
    print('iterate through param2 = ', jj)
    print('model timesteps for 1000 morphologic years = ', int(model._nt))
    print(param[0], '=', param1[ii])
    print(param[1], '=', param2[jj])
     
    # run the brie model
    for time_step in range(int(model._nt)-1):
        Brie.update(model)  # update the model by a time step
        
    # finalize by deleting variables and make Qinlet m^3/yr
    Brie.finalize(model)
        
    return model

#%%
###############################################################################
# run model
###############################################################################
for ii in inputs_p1:  
    for jj in inputs_p2 :
        model_out = batchBrie(ii, jj, param1, param2, param, name, mat)
        output[ ii, jj ] = model_out #([b_out, param1, param2, param])
    
#%%
###############################################################################
# save output
###############################################################################
       
filename = name+'.npz'
np.savez(filename, output=output)


# debugging the new inlet modifications from Jaap
# I think the differences in the indices inlet_idx (for the same shoreline and wave angle) between the matlab and python
# models come from rounding errors on x_s when imported into python (not enough precision), but this should be examined
# further as a rewrite the inlet model for Barrier3D
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
