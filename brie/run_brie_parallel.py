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

from joblib import Parallel, delayed
import multiprocessing
    
#%%
###############################################################################
# guts
###############################################################################

# subset of indices for testing grid discretization
name   = 'python_grid_testing_inlets_off'
param  = ['_dt','_dy'] 
param1 = [0.01, 0.02, 0.025, 0.05, 0.08, 0.1, 0.2, 0.25]
param2 = [1000, 800, 500, 400, 250, 100, 80, 50]

ii=[4, 1, 1, 8, 8]
jj=[6, 8, 1, 1, 8]
inputs = range(np.size(ii))

# import structures from Matlab
mat = loadmat('matlab_grid_testing_inlets_off_V7pt1.mat')
    
def batchBrie(kk, ii, jj, param1, param2, param, name, mat):
    
    # get matlab seed parameters
    xs = [mat['output'][ii[kk]-1][jj[kk]-1]['xs'].flat[0]]
    xs = np.r_[xs[0]].flatten()    # this is a numpy array - make sure that's ok
    wave_angle = [mat['output'][ii[kk]-1][jj[kk]-1]['wave_angle'].flat[0]]
    wave_angle = list(np.r_[wave_angle[0]].flatten())  # this is a list
    
    # create an instance of the pymt Brie model
    model = Brie()  # this initializes the model and calls __init__
    
    # update the initial conditions
    model._name = name
    model._plot_on = False
    model._make_gif = False
    model._inlet_model_on = False
    model._bseed = True
    model._xs = xs
    model._wave_angle = wave_angle
        
    setattr(model, param[0], param1[ii[kk]-1]) 
    setattr(model, param[1], param2[jj[kk]-1]) 
    
    print(param1[ii[kk]-1])
    print(param2[jj[kk]-1])
     
    # run the brie model
    for time_step in range(int(model._nt)-1):
        Brie.update(model)  # update the model by a time step
    
    return model

# check cores
num_cores = multiprocessing.cpu_count()
print("numCores = " + str(num_cores))

results = Parallel(n_jobs=num_cores)(delayed(batchBrie)(kk, ii, jj, param1, param2, param, name, mat) for kk in inputs)

#%%
###############################################################################
# save output
###############################################################################
       
filename = 'test2.npz'
np.savez(filename, results=results)