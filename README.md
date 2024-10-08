BRIE: The Barrier Inlet Environment model
=========================================

This is the python version of the BarrieR Inlet Envrionment (BRIE) model written
by Jaap Nienhuis and Jorge Lorenzo Trueba. The matlab to python conversion was
performed by Katherine Anarde. Eric Hutton worked to wrap the model with a basic
model interface, reorganized the many model components into classes (i.e., The
AlongshoreTransporter, InletSpinner), which now allows for dynamical testing of
different modules (see the Jupyter notebooks).

When using this BRIE model, please reference http://dx.doi.org/10.5194/gmd-2019-10.

###############################################################################
# Overview of model dynamics
Models that assess barrier island changes over geological timescales typically focus only on
storm overwash, which is more suited to a cross-sectional approach. However, several field
studies (Pierce, 1969, 1970) suggest that tidal inlets play a significant role in transgressive
sediment movement. The BRIE model expands upon the formulations of Lorenzo-Trueba and Ashton (2014)
by incorporating the alongshore direction and accounting for tidal inlet morphodynamics using 
Delft3D-derived parameterizations (Nienhuis and Ashton, 2016).
BRIE modeling is a framework to investigate barrier island responses to sea-level rise (SLR). 
This model incorporates longshore processes by connecting the cross-shore barrier island model
from LTA14 with a series of dynamic cross-shore profiles. Storm overwash and shoreface response 
functions are applied independently to each cell (see below figure). Alongshore sediment transport,
coupled with overwash dynamics, introduces feedback mechanisms that modify shoreline positions and
influence shoreface slopes and barrier overwash. The model also simulates the formation, closure, 
and migration of tidal inlets using parameterizations from NA16 (see below figure).
To our knowledge, this is the first long-term morphodynamic model (decadal to millennial timescales) 
for barrier island evolution that incorporates both tidal and overwash sediment fluxes.
![figure](/Users/rsahrae/PycharmProjects/pythonProject_CE_roya/brie_model.png)

###############################################################################

# add a similar overview of model dynamics as what is written below. you can pull from
# the original brie publication for some text (reworded). BUT, you want to include
# a statement about how the code is structurally different from the brie matlab version
# (including the waves class, alongshore transporter, and inlet spinner).




###############################################################################
## Default initial conditions
###############################################################################
```bash
barrier_model_on = True   # overwash and shoreface formulations on or off

ast_model_on     = True   # alongshore transport on or off

inlet_model_on   = False  # inlets on or off
```
###############################################################################
# wave climate parameters
###############################################################################
```bash
wave_height = 1     # mean offshore significant wave height [s],

wave_period = 10    # mean wave period [m],

wave_asym = 0.8     # fraction of waves approaching from left (looking onshore),

wave_high = 0.2     # " " from angles higher than 45 degrees from shore normal,
```
###############################################################################
# alongshore distribution of wave energy
###############################################################################
```bash
wave_climl = 180   # resolution of possible wave approach angles (1 per deg)

AngArray = np.linspace(-0.5*np.pi, 0.5*np.pi, wave_climl)

k for alongshore transport, from Nienhuis, Ashton, Giosan 2015 (Ashton 2006
 value for k is wrong)

k = 5.3e-06*1050*(g**1.5)*(0.5**1.2)*(np.sqrt(g*0.78)/(2*np.pi))**0.2
```
###############################################################################
# barrier model parameters
###############################################################################
```bash
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
```
|Build Status|


.. |Build Status| image:: https://travis-ci.com/UNC-CECL/brie.svg?branch=master
   :target: https://travis-ci.com/UNC-CECL/brie
###############################################################################
# Installation
###############################################################################

To install the latest release of brie using pip, simply run the following in your terminal of choice:

```bash
pip install brie
```
### From Source

*brie* is actively being developed on GitHub, where the code is freely available.
If you would like to modify code or contribute new code to *brie*, you will first
need to get *brie*'s source code, and then install *brie* from that code.

To get the source code you can either clone the repository with *git*:

```bash
git clone git@github.com:UNC-CECL/brie.git
```

or download a [zip-file](https://github.com/UNC-CECL/brie/archive/refs/heads/mater-brie.zip) :

```bash
curl -OL https://github.com/UNC-CECL/brie/archive/refs/heads/mater-brie.zip
```
Once you have a copy of the source code, you can install it into your current
environment,

```bash
pip install -e .
```

###############################################################################
# Brie Simulation
###############################################################################

```bash
def run_brie(n_steps, dt, dy, x_shoreline, wave_angle):
    brie = Brie(
        name=f"dt={dt},dy={dy}",
        bseed=True,
        wave_height=1.0,
        wave_period=7,
        barrier_width_critical=450.0,
        barrier_height_critical=1.9,
        alongshore_section_length=dy,
        time_step=dt,
        time_step_count=n_steps,
        wave_angle=wave_angle,
        xs=x_shoreline,
    )

    for _ in range(n_steps - 1):
        brie.update()  # update the model by a time step

    # finalize by deleting variables and make Qinlet m^3/yr
    brie.finalize()

    return brie
```

