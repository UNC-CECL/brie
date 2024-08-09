import numpy as np
import yaml

from .alongshore_transporter import AlongshoreTransporter
from .waves import ashton
from .inlet_spinner import InletSpinner
# from alongshore_transporter import AlongshoreTransporter
# from waves import ashton
# from lexi_inlet_spinner import InletSpinner


def inlet_fraction(a, b, c, d, I):
    """what are the inlet fractions"""
    return a + (b / (1 + c * (I ** d)))


SECONDS_PER_YEAR = 3600.0 * 24.0 * 365.0


class BrieError(Exception):
    pass


class Brie:
    def __init__(
        self,
        name="ExampleBarrierPlot5",
        barrier_model=True,
        ast_model=True,
        inlet_model=True,
        sed_strat=False,
        bseed=False,
        b3d=False,
        wave_height=1.0,
        wave_period=10,
        wave_asymmetry=0.8,
        wave_angle_high_fraction=0.2,
        wave_angle_resolution=1.0,
        sea_level_rise_rate=2e-3,
        sea_level_initial=10.0,
        barrier_width_critical=200.0,
        barrier_height_critical=2.0,
        max_overwash_flux=20,
        sea_water_density=1025.0,
        tide_amplitude=0.5,
        tide_frequency=1.4e-4,
        back_barrier_marsh_fraction=0.5,
        back_barrier_depth=3.0,
        lagoon_manning_n=0.05,
        shoreface_grain_size=2e-4,
        xshore_slope=1e-3,
        alongshore_section_length=100.0,
        alongshore_section_count=1000,
        time_step=0.05,
        time_step_count=100000,
        save_spacing=1e3,
        inlet_min_spacing=10000.0,
        wave_angle=None,
        xs=None,
    ):
        """The Barrier Inlet Environment model, BRIE.

        Parameters
        ----------
        name: string, optional
            Name of simulation
        barrier_model: bool, optional
            If `True`, use overwash and shoreface formulations.
        ast_model: bool, optional
            If `True`, turn on the alongshore transport model.
        inlet_model: bool, optional
            If `True`, turn on the inlets model.
        sed_strat: bool, optional
            If `True`, turn on the stratigraphy model (generate stratigraphy at a certain location).
        bseed: bool, optional
            If `True`, seed the model for comparison with matlab model.
        b3d: bool, optional
            If `True`, use Barrier3D as overwash model (barrier_model must be False).
        wave_height: float, optional
            Mean offshore significant wave height [m].
        wave_period: float, optional
            Mean wave period [s].
        wave_asymmetry: float, optional
            Fraction of waves approaching from left (looking onshore).
        wave_angle_high_fraction: float, optional
            Fraction of waves approaching from angles higher than 45 degrees.
        wave_angle_resolution: float, optional
            Resolution of possible wave approach angles, typically 1 per degree [deg].
        sea_level_rise_rate: float, optional
            Rate of sea_level rise [m/yr].
        sea_level_initial: float, optional
            Initial sea level elevation [m]
        barrier_width_critical: float, optional
            Critical barrier width [m].
        barrier_height_critical: float, optional
            Critical barrier height [m].
        max_overwash_flux: float, optional
            Maximum overwash flux [m3/m/yr].
        sea_water_density: float, optional
            Density of sea water [kg/m^3].
        tide_amplitude: float, optional
            Amplitude of tide [m].
        tide_frequency: float, optional
            Tidal frequency [rad/s].
        back_barrier_marsh_fraction: float, optional
            Percent of backbarrier covered by marsh and does not contribute to tidal prism.
        back_barrier_depth: float, optional
            Depth of the back barrier [m].
        lagoon_manning_n: float, optional
            Bulk manning n of the lagoon [s m^-(1/3)].
        shoreface_grain_size: float, optional
            Median grain size of the shoreface [m].
        xshore_slope: float, optional
            Background cross-shore slope (beta) [-].
        alongshore_section_length: float, optional
            Length of each alongshore section [m].
        alongshore_section_count: int, optional
            Number of alongshore sections.
        time_step: float, optional
            Timestep of the numerical model [y].
        time_step_count: int, optional
            Number of time steps.
        save_spacing: int, optional
            Saving interval.
        inlet_min_spacing: float, optional
            Minimum inlet spacing [m].
        wave_angle: float, optional
            An array of wave angles for seeding (remove stochasticity).
        xs: float, optional
            An array of shoreline position for seeding (remove stochasticity).

        Examples
        --------
        >>> from brie import Brie
        >>> brie = brie()
        """

        # name of output file
        self._name = name

        # which modules to run
        self._barrier_model_on = barrier_model
        self._ast_model_on = ast_model
        self._inlet_model_on = inlet_model
        self._sedstrat_on = sed_strat
        self._bseed = bseed
        self._b3d_barrier_model_on = b3d

        if self._b3d_barrier_model_on is True and self._barrier_model_on is True:
            raise ValueError(" Please set only one barrier model to 'True' ")

        # general parameters
        self._rho_w = sea_water_density
        # self._g = scipy.constants.g
        self._g = (
            9.81  # KA: above has too much precision for comparison to Matlab version
        )

        ###############################################################################
        # wave climate
        ###############################################################################

        self._wave_height = wave_height
        self._wave_period = wave_period
        self._wave_asym = wave_asymmetry
        self._wave_high = wave_angle_high_fraction
        self._wave_angle = wave_angle  # the default initial wave angle

        # asthon wave distribution from waves (needed for inlet spinner and AST!!!)
        self._wave_dist = ashton(a=self._wave_asym, h=self._wave_high, loc=-np.pi / 2, scale=np.pi)

        ###############################################################################
        # barrier model parameters
        ###############################################################################

        self._slr = [
            sea_level_rise_rate
        ] * int(time_step_count)  # KA: made this a TS so I can replace with accelerated SLR
        self._s_background = xshore_slope
        self._w_b_crit = barrier_width_critical
        self._h_b_crit = barrier_height_critical
        self._Qow_max = max_overwash_flux
        self._z = sea_level_initial
        self._bb_depth = back_barrier_depth
        self._grain_size = shoreface_grain_size
        self._R = 1.65  # relative density of sand
        self._e_s = 0.01  # suspended sediment transport efficiency factor
        self._c_s = 0.01  # shoreface transport friction factor

        # alongshore grid setup
        self._dy = alongshore_section_length
        self._ny = alongshore_section_count

        # time stepping
        self._time_index = 1
        self._dt = time_step
        self._nt = time_step_count
        self._dtsave = save_spacing

        # boolean for drowning
        self._drown = False

        ###############################################################################
        # inlet model parameters & functions
        ###############################################################################

        self._Jmin = inlet_min_spacing
        self._a0 = tide_amplitude
        self._omega0 = tide_frequency
        self._man_n = lagoon_manning_n
        self._marsh_cover = back_barrier_marsh_fraction
        self._RNG = np.random.default_rng(seed=1973)  # random number generator


        ###############################################################################
        # shoreface and barrier model dependent variables
        ###############################################################################

        self._Vd_max = self._w_b_crit * self._h_b_crit  # max deficit volume [m3/m]

        w_s = (
            self._R
            * self._g
            * self._grain_size ** 2
            / (
                (18 * 1e-6)
                + np.sqrt(0.75 * self._R * self._g * (self._grain_size ** 3))
            )
        )  # settling velocity [m/s] Church & Ferguson (2004)

        phi = (
            16 * self._e_s * self._c_s / (15 * np.pi * self._R * self._g)
        )  # phi from Ortiz and Ashton (2016)

        self._z0 = (
            2 * self._wave_height / 0.78
        )  # minimum depth of integration [m] (simple approx of breaking wave depth based on offshore wave height)

        self._d_sf = (
            8.9 * self._wave_height
        )  # depth shoreface [m], Hallermeier (1983) or  Houston (1995)
        # alternatively 0.018*wave_height*wave_period*sqrt(g./(R*grain_size))

        self._k_sf = (
            (3600 * 24 * 365)
            / (self._d_sf - self._z0)
            * (
                self._g ** (15 / 4)
                * self._wave_height ** 5
                * phi
                * self._wave_period ** (5 / 2)
                / (1024 * np.pi ** (5 / 2) * w_s ** 2)
                * (4 / 11 * (1 / self._z0 ** (11 / 4) - 1 / (self._d_sf ** (11 / 4))))
            )
        )  # shoreface response rate [m^3/m/yr], Lorenzo-Trueba & Ashton (2014) or Ortiz and Ashton (2016)

        self._s_sf_eq = (
            3
            * w_s
            / 4
            / np.sqrt(self._d_sf * self._g)
            * (5 + 3 * self._wave_period ** 2 * self._g / 4 / (np.pi ** 2) / self._d_sf)
        )  # equilibrium shoreface slope

        self._x_t = (self._z - self._d_sf) / self._s_background + np.zeros(
            self._ny
        )  # position shoreface toe [m]

        # KA - used for testing/debugging brie_org.py vs. brie.m
        if self._bseed:
            if xs is None or wave_angle is None:
                raise ValueError("if bseed is True, xs and wave_angle must be provided")
            self._x_s = xs
            # self._wave_angle = wave_angle  # this is redundant
        else:
            self._x_s = (
                self._RNG.random(self._ny) + self._d_sf / self._s_sf_eq + self._x_t
            )  # position shoreline [m]
            # self._x_s = np.random.rand(self._ny) + self._d_sf / self._s_sf_eq + self._x_t  # position shoreline [m]

        self._x_b = (
            self._d_sf / self._s_sf_eq + self._w_b_crit + self._x_t
        )  # position back barrier [m]

        self._h_b = 2 + np.zeros(self._ny)  # height barrier [m]

        # initialize empty arrays for barrier model (added by KA for coupling)
        self._x_t_dt = np.zeros(self._ny)
        self._x_b_dt = np.zeros(self._ny)
        self._x_s_dt = np.zeros(self._ny)
        self._h_b_dt = np.zeros(self._ny)
        self._x_b_fld_dt = 0

        ###############################################################################
        # inlet model dependent variables
        ###############################################################################
        #The basin_width calculation is commented out and moved to the correct place to be updated at each iteration.
        # self._basin_width = np.maximum(0, self._z / self._s_background - self._x_b)
        # initialize inlet module
        self._inlets = InletSpinner(
            self._x_s,
            self._x_b,
            sea_level=self._z,
            back_barrier_depth=self._bb_depth,
            xshore_slope=self._s_background,
            sea_water_density=self._rho_w,
            barrier_width_critical=self._w_b_crit,
            number_time_steps=self._nt,
            save_spacing=self._dtsave,
            basin_width=np.maximum(0, self._z / self._s_background - self._x_b), #initial value of basin width given inline
            inlet_storm_frequency=10,
            barrier_height=self._h_b,
            alongshore_section_length=self._dy,
            time_step=self._dt,
            wave_height=self._wave_height,
            wave_period=self._wave_period,
            wave_angle=self._wave_angle,
            wave_distribution=self._wave_dist,
            inlet_min_spacing=self._Jmin,
            tide_amplitude=self._a0,
            tide_frequency=self._omega0,
            lagoon_manning_n=self._man_n,
            back_barrier_marsh_fraction=self._marsh_cover,
        )
        # self._basin_width = np.maximum(0, self._z / self._s_background - self._inlets.bay_shoreline_x)
        # moved from the inlet model on section of master BRIE to outside the inlet class
        # but it is the same calculation for basin width
        ###############################################################################
        # Alongshore sediment transport model dependent variables (added)
        ###############################################################################

        # initialize the alongshore transport model
        self._transporter = AlongshoreTransporter(
            self._x_s,
            alongshore_section_length=self._dy,
            time_step=self._dt,
            change_in_shoreline_x=self._x_s_dt,
            wave_height=self._wave_height,
            wave_period=self._wave_period,
            wave_angle=self._wave_angle,
            wave_distribution=self._wave_dist,
        )

        ###############################################################################
        # variables used for saving data
        ###############################################################################

        self._t = np.arange(
            self._dt, (self._dt * self._nt) + self._dt, self._dt
        )  # time array
        self._Qoverwash = np.float32(np.zeros(int(self._nt)))  # overwash flux [m^3/yr]
        self._Qshoreface = np.float32(
            np.zeros(int(self._nt))
        )  # KA: new variable for time series of shoreface flux [m^3/yr]
        self._Qinlet = np.float32(np.zeros(int(self._nt)))  # inlet flux [m^3/yr]

        # KA - added these back after Eric's rewrite because I needed them for testing
        c_idx = np.uint8(np.zeros((self._ny, 1000)))  # noqa: F841
        bar_strat_x = (  # noqa: F841
            self._x_b[0] + 1000
        )  # cross-shore location where to record stratigraphy. I guess would be better to do it at one instant
        # in time rather than space?
        self._x_t_save = np.int32(
            np.zeros((self._ny, np.size(np.arange(0, self._nt, self._dtsave))))
        )
        self._x_t_save[
            :, 0
        ] = (
            self._x_t
        )  # KA: for some reason this rounds down to 1099 and not up to 1100...why?
        self._x_s_save = np.int32(
            np.zeros((self._ny, np.size(np.arange(0, self._nt, self._dtsave))))
        )
        self._x_s_save[:, 0] = self._x_s
        self._x_b_save = np.int32(
            np.zeros((self._ny, np.size(np.arange(0, self._nt, self._dtsave))))
        )
        self._x_b_save[:, 0] = self._x_b
        self._h_b_save = np.float32(
            np.zeros((self._ny, np.size(np.arange(0, self._nt, self._dtsave))))
        )
        self._h_b_save[:, 0] = self._h_b
        self._s_sf_save = np.float32(
            np.zeros((self._ny, np.size(np.arange(0, self._nt, self._dtsave))))
        )
        # self._s_sf_save[:, 0] = self._s_sf_eq
        self._s_sf_save[:, 0] = self._d_sf / (
            self._x_s - self._x_t
        )  # added by KA to represent the actual initial s_sf


    @classmethod
    def from_yaml(cls, filepath):
        with open(filepath, "r") as fp:
            params = yaml.safe_load(fp)
        return cls(**params)

    @property
    def time_index(self):
        return self._time_index

    @property
    def time_step(self):
        return self._dt

    @property
    def time(self):
        return self.time_index * self.time_step

    @property
    def nt(self):
        return self._nt

    @property
    def drown(self):
        return self._drown

    @property
    def x_t_dt(self):
        return self._x_t_dt

    @x_t_dt.setter
    def x_t_dt(self, value):
        self._x_t_dt = value

    @property
    def x_s_dt(self):
        return self._x_s_dt

    @x_s_dt.setter
    def x_s_dt(self, value):
        self._x_s_dt = value

    @property
    def x_b_dt(self):
        return self._x_b_dt

    @x_b_dt.setter
    def x_b_dt(self, value):
        self._x_b_dt = value

    @property
    def h_b_dt(self):
        return self._h_b_dt

    @h_b_dt.setter
    def h_b_dt(self, value):
        self._h_b_dt = value

    @property
    def x_t(self):
        return self._x_t

    @property
    def x_s(self):
        return self._x_s

    @property
    def x_b(self):
        return self._x_b

    @x_b.setter
    def x_b(self, value):
        self._x_b = value

    @property
    def x_b_save(self):
        return self._x_b_save

    @x_b_save.setter
    def x_b_save(self, value):
        self._x_b_save = value

    @property
    def h_b(self):
        return self._h_b

    @h_b.setter
    def h_b(self, value):
        self._h_b = value

    @property
    def h_b_save(self):
        return self._h_b_save

    @h_b_save.setter
    def h_b_save(self, value):
        self._h_b_save = value

    @property
    def slr(self):
        return self._slr

    @slr.setter
    def slr(self, value):
        self._slr = value

    @property
    def ny(self):
        return self._ny

    @property
    def d_sf(self):
        return self._d_sf

    @property
    def k_sf(self):
        return self._k_sf

    @property
    def s_sf_eq(self):
        return self._s_sf_eq

    @property
    def wave_angle(self):
        return self._wave_angle

    @wave_angle.setter
    def wave_angle(self, new_angle):
        if new_angle > 90.0 or new_angle < -90:
            raise ValueError("wave angle must be between -90 and 90 degrees")
        self._wave_angle = new_angle


    def update(self):
        """Update BRIE by a single time step."""
        self._time_index += 1
        # print('time_index=',self._time_index)

        # sea level
        self._z = self._z + (
            self._dt * self._slr[self._time_index - 1]
        )  # height of sea level
        w = self._x_b - self._x_s  # barrier width
        d_b = np.minimum(
            self._bb_depth * np.ones(np.size(self._x_b)),
            self._z - (self._s_background * self._x_b),
        )  # basin depth
        s_sf = self._d_sf / (self._x_s - self._x_t)  # shoreface slope

        # if the barrier drowns, break
        if np.sum(w < -10) > (self._ny / 2) or np.any(w < -1000):
            self._drown = True
            print("Barrier Drowned - break")


        if self._barrier_model_on:
            # volume deficit
            Vd_b = np.maximum(0, (self._w_b_crit - w) * (self._h_b + d_b))
            Vd_h = np.maximum(0, (self._h_b_crit - self._h_b) * w)
            Vd = Vd_b + Vd_h

            # overwash fluxes [m^3/m]
            Qow_b = self._dt * self._Qow_max * Vd_b / np.maximum(Vd, self._Vd_max)
            # overwash flux deposited in the back barrier (LTA14)
            Qow_h = self._dt * self._Qow_max * Vd_h / np.maximum(Vd, self._Vd_max)
            # overwash flux deposited on top of the existing barrier (LTA14)
            Qow = Qow_b + Qow_h

            # shoreface flux [m^3/m]
            Qsf = self._dt * self._k_sf * (self._s_sf_eq - s_sf)

            # changes
            ff = (self._z - self._s_background * self._x_b - d_b) / (
                self._z - self._s_background * self._x_b + self._h_b
            )
            self._x_t_dt = (
                4
                * Qsf
                * (self._h_b + self._d_sf)
                / (self._d_sf * (2 * self._h_b + self._d_sf))
            ) + (2 * self._dt * self._slr[self._time_index - 1] / s_sf)
            self._x_s_dt = 2 * Qow / ((2 * self._h_b) + self._d_sf) / (1 - ff) - (
                4
                * Qsf
                * (self._h_b + self._d_sf)
                / (((2 * self._h_b) + self._d_sf) ** 2)
            )
            self._x_b_dt = Qow_b / (self._h_b + d_b)
            self._h_b_dt = (Qow_h / w) - (self._dt * self._slr[self._time_index - 1])

            # how much q overwash w in total [m3/yr]
            self._Qoverwash[self._time_index - 1] = np.sum(self._dy * Qow_b / self._dt)

            # how much q shoreface in total [m3/yr] [KA: added for comparison to B3D]
            self._Qshoreface[self._time_index - 1] = np.sum(self._dy * Qsf / self._dt)

        elif self._b3d_barrier_model_on:
            pass
            # do nothing, x_t_dt, x_s_dt, x_b_dt, and h_b_dt all come from Barrier3d (is there a better way to do this?)
            # self._x_t_dt = self._x_t_dt
            # self._x_s_dt = self._x_s_dt
            # self._x_b_dt = self._x_b_dt
            # self._h_b_dt = self._h_b_dt

        else:
            self._x_t_dt = np.zeros(self._ny)
            self._x_s_dt = np.zeros(self._ny)
            self._x_b_dt = np.zeros(self._ny)
            self._h_b_dt = np.zeros(self._ny)

        if self._inlet_model_on:

            if self._bseed:
                self._inlets.wave_angle = self._wave_angle[self._time_index-1]

            # update inlet model parameters prior to advancing one TS
            self._inlets.shoreline_x = self._x_s
            self._inlets.bay_shoreline_x = self._x_b
            self._inlets._x_s_dt = self._x_s_dt
            self._inlets.update(self._h_b, self._z)
            #self._x_s_dt = self._inlets._x_s_dt
            self._x_b_fld_dt = self._inlets._x_b_fld_dt #get the updated values from inlet module
            self._Qinlet = self._inlets._Qinlet


        else:  # inlet model not on (commented out sections were already commented)
            # Qs_in = 0
            # delta = 0
            # delta_r = 0
            # inlet_sink = 0
            self._x_b_fld_dt = 0

        # do implicit thing (updated on May 27, 2020 to force shoreline diffusivity to be greater than zero)
        # copied from KA brie
        # I think we will want to put this before inlets, OR we just need to use the necessary functions
        # in inlet spinner coming from ast
        if self._ast_model_on:

            # update AlongshoreTransporter with new change in shoreline position
            # RS, wrong variable was updated, commented out , we performed the update inside the function
            # self._transporter.dx_dt = (
            #     self._x_s_dt
            # )  # DO I ALSO NEED TO PROVIDE SHORELINE X?

            self._transporter.update(self._x_s_dt, self._x_s)
            self._x_s = self._transporter.shoreline_x  # new shoreline position

        else:
            self._x_s = self._x_s + self._x_s_dt

        # how are the other moving boundaries changing?
        self._x_t = self._x_t + self._x_t_dt
        self._x_b = self._x_b + self._x_b_dt + self._x_b_fld_dt
        self._h_b = self._h_b + self._h_b_dt

        # save subset of BRIE variables
        # (KA: I changed this from mod = 1 to mod = 0 to allow for saving every 1 timestep)
        """save subset of BRIE variables"""
        if np.mod(self._time_index, self._dtsave) == 0:
            self._x_t_save[
                :, np.fix(self._time_index / self._dtsave).astype(int) - 1
            ] = self._x_t
            self._x_s_save[
                :, np.fix(self._time_index / self._dtsave).astype(int) - 1
            ] = self._x_s
            self._x_b_save[
                :, np.fix(self._time_index / self._dtsave).astype(int) - 1
            ] = self._x_b
            self._h_b_save[
                :, np.fix(self._time_index / self._dtsave).astype(int) - 1
            ] = self._h_b
            self._s_sf_save[
                :, np.fix(self._time_index / self._dtsave).astype(int) - 1
            ] = s_sf

    ###############################################################################
    # Finalize: only return necessary variables
    ###############################################################################

    def finalize(self):
        if self._inlet_model_on:
            self._Qinlet = self._Qinlet / self._dt  # put into m3/yr
            self._Qinlet_norm = (self._Qinlet / self._dy)  # put into m3/m/yr



if __name__ == "__main__":
    from brie import Brie
    brie = Brie()

    for i in range(50):
        print(brie.drown)
        print(brie.time_step)
        brie.update()