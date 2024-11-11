BRIE: The Barrier Inlet Environment model
=========================================

This is the python version of the BarrieR Inlet Envrionment (*brie*) model written
by Jaap Nienhuis and Jorge Lorenzo Trueba. The matlab to python conversion was
performed by Katherine Anarde. Eric Hutton worked to wrap the model with a basic
model interface. Lexi Van Blunk and Roya Sahraei reorganized model 
components into classes (i.e., The AlongshoreTransporter, InletSpinner), which 
now allows for dynamical testing of different modules (see the Jupyter notebooks)
and easier coupling with other modeling frameworks.

When using any version of the *brie* model, please reference the original publication:
http://dx.doi.org/10.5194/gmd-2019-10.

###############################################################################
# BRIE Model
Models that simulate barrier island changes over geological timescales (centuries to millenia) 
typically focus on the role of storm overwash, and its couplings with shoreface and back-barrier dynamics. 
However, field studies suggest that tidal inlets play a significant role in transgressive
sediment movement. The *brie* model was developed to test this hypothesis. 
*brie* combines the cross-shore barrier evolution formulations of Lorenzo-Trueba and Ashton (2014) with 
tidal inlet morphodynamics described by the Delft3D-derived parameterizations of Nienhuis and Ashton (2016). 
Storm overwash and shoreface response functions from Lorenzo-Trueba and Ashton (2014) are applied independently 
to each cross-sectional transect. Cross-shore transects are connected in the alongshore via the diffusive sediment transport 
model of Ashton and Murray (2006). Transects are converted to inlets based on hydro- and morpho-dynamic 
criteria; thereafter, inlets can migrate, merge, and close. *brie* is the first long-term morphodynamic model for 
barrier island evolution that incorporates both 
tidal and overwash sediment fluxes. For a complete description
of model dynamics, the reader is directed to http://dx.doi.org/10.5194/gmd-2019-10.

In this python version of *brie*, there is functionality to alternatively use Barrier3D
(Reeves et al., 2019) as the barrier overwash model. Coupling with Barrier3D requires the model CASCADE 
(Anarde et al., 2024a,b). 
![figure](/notebooks/brie_domain.png)

###############################################################################
# Installation
###############################################################################

To install the latest release of brie using pip, simply run the following in your terminal of choice:

```bash
pip install brie .
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
conda install baked-brie
```
###############################################################################
# Run scripts
###############################################################################
Here we describe the basic commands to import, initialize, and iteratively update the *brie* model class for each
timestep.

**First, import the brie model**
```bash
from brie.brie import Brie
```

**Initialize a model using the default conditions**
```bash
brie = Brie()
```
For a complete list of default conditions, the reader is directed to Nienhuis et al., 2019. 
Alternatively, in this python version of *brie*, the variables are defined in the initialization function 
(i.e., in _init_ ).

**Call the *update* function to advance one time step**
```bash
brie.update()
```

**Alternatively, the *update* function can be called within a loop**
```bash
for i in range(brie.nt-1):
    if brie.drown:
        break
    brie.update() 
```

**Below is an example of a model initialization with initial conditions other than default:**
```bash
brie = Brie(
        wave_height=1.0,  # Mean offshore significant wave height [m]
        wave_period=15,  # Mean wave period [s]
        sea_level_rise_rate=4e-2,  # Rate of sea_level rise [m/yr]
        barrier_width_critical=150.0,  # Critical barrier width [m]
        barrier_height_critical=1.5,  # Critical barrier height [m]
        max_overwash_flux=40,  # Maximum overwash flux [m3/m/yr]
        tide_amplitude=1.0,  # Amplitude of tide [m]
        back_barrier_marsh_fraction=0.6,  # Percent of backbarrier covered by marsh and does not contribute to tidal prism
        back_barrier_depth=2.0,  # Depth of the back barrier [m]
        shoreface_grain_size=1e-4,  # Median grain size of the shoreface [m]
        alongshore_section_length=200.0,  # Length of each alongshore section [m]
        alongshore_section_count=2000,  # Number of alongshore sections
        time_step=1,  # Timestep of the numerical model [y]
        time_step_count=2000,  # Number of time steps
        save_spacing=1e2,  # Saving interval
        inlet_min_spacing=50000.0,  # Minimum inlet spacing [m]
)
```

**The *finalize* function deletes some variables to make a smaller file for saving. 
It also modifies the variable Qinlet to have units of m3/m/yr instead of m^3/yr**
```bash 
brie.finalize()       
```

###############################################################################
# References
###############################################################################
