BRIE: The Barrier Inlet Environment model
=========================================
###############################################################################
# Input conditions
###############################################################################

barrier_model_on = True   # overwash and shoreface formulations on or off

ast_model_on     = True   # alongshore transport on or off

inlet_model_on   = False  # inlets on or off

###############################################################################
# wave climate parameters
###############################################################################

wave_height = 1     # mean offshore significant wave height [s],

wave_period = 10    # mean wave period [m],

wave_asym = 0.8     # fraction of waves approaching from left (looking onshore),

wave_high = 0.2     # " " from angles higher than 45 degrees from shore normal,

# alongshore distribution of wave energy

wave_climl = 180   # resolution of possible wave approach angles (1 per deg)

AngArray = np.linspace(-0.5*np.pi, 0.5*np.pi, wave_climl)

# k for alongshore transport, from Nienhuis, Ashton, Giosan 2015 (Ashton 2006
# value for k is wrong)
k = 5.3e-06*1050*(g**1.5)*(0.5**1.2)*(np.sqrt(g*0.78)/(2*np.pi))**0.2 

###############################################################################
# barrier model parameters
###############################################################################

slr = 2e-03            # sea level rise rate [m/yr]

s_background = 1e-03   # background cross-shore slope (beta)

w_b_crit = 200         # critical barrier width [m]

h_b_crit = 2           # critical barrier height [m]

Qow_max = 20           # max overwash flux [m3/m/yr]

z = 10                 # initial sea level [m]

bb_depth = 3           # back barrier depth [m]

grain_size = 2e-04     # median grain size of the shoreface [m]

R = 1.65               # relative density of sand

e_s = 0.01             # suspended sediment tranport efficiency factor

c_s = 0.01             # shoreface transport friction factor

# alongshore grid setup
dy = 100               # length of each alongshore section [m]

ny = 1000              # number of alonghsore sections

# timestepping
dt = 0.05              # timestep of the numerical model [years]

nt = 1e4#1e5           # number of timesteps

dtsave = 1e3           # save spacing
|Build Status|


.. |Build Status| image:: https://travis-ci.com/UNC-CECL/brie.svg?branch=master
   :target: https://travis-ci.com/UNC-CECL/brie
