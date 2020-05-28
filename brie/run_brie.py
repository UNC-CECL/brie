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
name   = 'python_grid_testing_inlets_on_full_v5'
param  = ['_dt','_dy'] 
param1 = [0.01, 0.02, 0.025, 0.05, 0.08, 0.1, 0.2, 0.25]
param2 = [1000, 800, 500, 400, 250, 100, 80, 50]  

inputs_p1 = range(np.size(param1))
inputs_p2 = range(np.size(param2))

# import structures from Matlab
mat = loadmat('matlab_grid_testing_inlets_on_V7pt1_v5_full.mat')

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
    model._bseed = True    
    setattr(model, param[0], param1[ii]) # dt
    setattr(model, param[1], param2[jj]) # dy
    model._nt = 1e3/param1[ii]  # timesteps for 1000 morphologic years
    
    # v5 - test against Jaap's paper Figure 9 initial parameters
    model._a0  = 1.5    # higher tidal amplitude (0.5 m)
    model._slr = 3e-3   # higher slr (2e-3 m/yr)
    model._wave_height = 1.5  # higher wave height (1 m)
    model._wave_period = 8    # lower wave period (8 s)
    
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
        
    # finalize by deleting variables
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