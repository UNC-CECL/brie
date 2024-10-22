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
# BRIE Model
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

# Overview of model dynamics
The BRIE model incorporates many dynamics and processes that influence 
barrier island evolution. In an earlier version of BRIE, all these 
processes were classes contained in a single file. However, the model was later
modified to separate these processes into individual modules, 
allowing easier coupling with other models. For instance, 
the processes are now written in separate Python files, and each must
be imported individually to be used.

**Overwash**: this process is a critical processes on barrier islands
which transportis material from the ocean side to the back-barrier lagoon. Overwash make changes to
height and width of the barrier by depositing sediments which is critical for barrier islands to 
migrate landward in response to sea-level rise. 

**Alongshore Transport Module**: The model accounts for the movement of sediment along the shoreline, driven by wave 
action. Alongshore sediment transport is essential for maintaining the barrier island's shape and adjusting its 
position over time.

**Waves Module**:

**Inlets Module**: Tidal inlets play a major role in the morphodynamics of barrier islands by influencing sediment
flux and barrier migration. The BRIE model allows for the formation, migration, merging, and closing of tidal inlets, 
depending on hydrodynamic conditions.

-Inlet Formation and Hydrodynamics: The model simulates the creation of new inlets when storm events breach the barrier 
island. Inlet characteristics such as width, depth, and flow velocity are calculated based on tidal prism and wave-driven 
sediment transport.

-Flood-Tidal Delta Deposition: Inlets contribute to landward sediment flux by depositing sediment in the back-barrier 
lagoon as flood-tidal deltas. This process helps the barrier island migrate landward, particularly when overwash alone
is insufficient to keep pace with sea-level rise.

-Inlet Migration: Inlets can migrate along the barrier island, primarily driven by the littoral drift. This migration 
redistributes sediment along the coastline and can modify the shape and position of the barrier over time.

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

```bash
curl -OL https://github.com/UNC-CECL/brie/archive/refs/heads/mater-brie.zip
```
Once you have a copy of the source code, you can install it into your current
environment,

```bash
pip install -e .
```
You can also use `conda`:

```bash
conda install coastal-brie
```
###############################################################################
# Example BRIE Simulations
###############################################################################

**First, import the the brie model**
```bash
from brie.brie import Brie
```

**Initialize a model using the default conditions**
```bash
brie = Brie(name="default")
```
For a complete list of default conditions, the reader is directed to Nienhuis et al., 2019. 
In this python version of brie, the variables are defined in the initialization function 
(brie._init_ function).

**Call the update function to advance one time step**
```bash
brie.update()
```

**If you want to initialize a model with inital conditions other than default:**
```bash

brie = Brie(
wave_height=1.0,
        wave_period=15,  # mean wave period [s]
        sea_level_rise_rate=4e-2,  # Rate of sea_level rise [m/yr]
        barrier_width_critical=150.0, # Critical barrier width [m]
        barrier_height_critical=1.5, #Critical barrier height [m]
        max_overwash_flux=40, #Maximum overwash flux [m3/m/yr]
        tide_amplitude=1.0, #Amplitude of tide [m]
        back_barrier_marsh_fraction=0.6, #Percent of backbarrier covered by marsh and does not contribute to tidal prism
        back_barrier_depth=2.0, # Depth of the back barrier [m]
        shoreface_grain_size=1e-4, #Median grain size of the shoreface [m]
        alongshore_section_length=200.0, #Length of each alongshore section [m]
        alongshore_section_count=2000, #Number of alongshore sections
        time_step=1, #Timestep of the numerical model [y]
        time_step_count=2000, # Number of time steps
        save_spacing=1e2, #Saving interval
        inlet_min_spacing=50000.0, # Minimum inlet spacing [m]
)

for _ in range(brie.nt - 1):
  brie.update()  # update the model by a time step
```

**The "finalize" function deletes some variables to make a smaller file for saving, 
and it makes the variable Qinlet m^3/yr instead of XXXXX**
```bash 
brie.finalize()       
```