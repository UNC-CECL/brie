import numpy as np
import scipy.constants
import yaml
from numpy.lib.scimath import power as cpower, sqrt as csqrt
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


from .alongshore_transporter import calc_alongshore_transport_k

def inlet_fraction(self, a, b, c, d, I):
    """what are the inlet fractions"""
    return a + (b / (1 + c * (I ** d)))


SECONDS_PER_YEAR = 3600.0 * 24.0 * 365.0


class Brie:
    def __init__(
        self,
        barrier_model=True,
        ast_model=True,
        inlet_model=True,
        wave_height=1.0,
        wave_period=10,
        wave_asymmetry=0.8,
        wave_angle_high_fraction=0.2,
        wave_angle_resolution=1.0,
        sea_level_rise_rate=2e-3,
        sea_level_initial=10.0,
        barrier_width_critical=200.0,
        barrier_height_critical=2.0,
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
        inlet_min_spacing=10000.0,
    ):
        """The Barrier Inlet Environment model, BRIE.

        Parameters
        ----------
        barrier_model: bool, optional
            If `True`, use overwash and shoreface formulations.
        ast_model: bool, optional
            If `True`, turn on the alongshore transport model.
        inlet_model: bool, optional
            If `True`, turn on the inlets model.
        wave_height: float, optional
            Mean offshore significant wave height [m].
        wave_period: float, optional
            Mean wave period [s].
        wave_asymmetry: float, optional
            Fraction of waves approaching from left (looking onshore).
        wave_angle_high_fraction: float, optional
            Fraction of waves approaching from angles higher than 45 degrees.
        wave_angle_resolution: float, optional
            Resolution of possible wave approach angles [deg].
        sea_level_rise_rate: float, optional
            Rate of sea_level rise [m/yr].
        sea_level_initial: float, optional
            Initial sea level elevation [m]
        barrier_width_critical: float, optional
            Critical barrier width [m].
        barrier_height_critical: float, optional
            Critical barrier height [m].
        sea_water_density: float, optional
            Density of sea water [kg/m^3].
        tide_amplitude: float, optional
            Amplitude of tide [m].
        tide_frequence: float, optional
            Tidal frequency [rad/s].
        back_barrier_marsh_fraction: float, optional
            Percent of backbarrier covered by marsh and therefore does not
            contribute to tidal prism.
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
        inlet_min_spacing: float, optional
            Minimum inlet spacing [m].

        Examples
        --------
        >>> from brie import Brie
        >>> brie = Brie()
        """
        # name of output file
        self._name = "ExampleBarrierPlot5"

        # which modules to run
        self._barrier_model_on = barrier_model
        self._ast_model_on = ast_model
        self._inlet_model_on = inlet_model
        self._sedstrat_on = False  # generate stratigraphy at a certain location
        self._bseed = False  # KA: used for testing discretization
        self._b3d_barrier_model_on = False  # KA: added this because I wasn't sure how else to pass the data from B3D

        # general parameters
        self._rho_w = sea_water_density
        self._g = scipy.constants.g

        ###############################################################################
        # wave climate parameters
        ###############################################################################

        self._wave_height = wave_height
        self._wave_period = wave_period
        self._wave_asym = wave_asymmetry
        self._wave_high = wave_angle_high_fraction
        self._wave_angle = 0.0  # the default initial wave angle

        # alongshore distribution of wave energy
        self._wave_climl = int(180.0 / wave_angle_resolution)
        self._angle_array = np.deg2rad(np.linspace(-90.0, 90.0, self._wave_climl))

        # self._wave_climl = (
        #     180  # resolution of possible wave approach angles (1 per deg)
        # )
        # self._angle_array = np.linspace(-0.5 * np.pi, 0.5 * np.pi, self._wave_climl)

        # k for alongshore transport, from Nienhuis, Ashton, Giosan 2015 (Ashton 2006
        # value for k is wrong)
        self._k = calc_alongshore_transport_k(gravity=self._g)

        ###############################################################################
        # barrier model parameters
        ###############################################################################

        self._slr = sea_level_rise_rate
        self._s_background = xshore_slope
        self._w_b_crit = barrier_width_critical
        self._h_b_crit = barrier_height_critical
        self._Qow_max = 20  # max overwash flux [m3/m/yr]
        self._z = sea_level_initial
        self._bb_depth = back_barrier_depth
        self._grain_size = shoreface_grain_size
        self._R = 1.65  # relative density of sand
        self._e_s = 0.01  # suspended sediment tranport efficiency factor
        self._c_s = 0.01  # shoreface transport friction factor

        # alongshore grid setup
        self._dy = alongshore_section_length
        self._ny = alongshore_section_count

        # timestepping
        self._dt = time_step
        self._nt = time_step_count
        self._dtsave = 1e3  # save spacing

        ###############################################################################
        # inlet model parameters & functions
        ###############################################################################

        self._Jmin = inlet_min_spacing
        self._a0 = tide_amplitude
        self._omega0 = tide_frequency
        self._inlet_asp = np.sqrt(0.005)  # aspect ratio inlet
        self._man_n = lagoon_manning_n
        self._u_e = 1  # inlet equilibrium velocity [m/s]
        self._inlet_max = 100  # maximum number of inlets (mostly for debugging)
        # % of backbarrier covered by marsh and therefore
        # does not contribute to tidal prism
        self._marsh_cover = back_barrier_marsh_fraction

        self.dependent()

    @classmethod
    def from_yaml(cls, filepath):
        with open(filepath, "r") as fp:
            params = yaml.safe_load(fp)
        return cls(**params)

    def dependent(self, wave_angle=None, xs=None):
        """Set the internal variables that depend on the input parameters.

        Parameters
        ----------
        wave_angle: float, optional
            If provided, the current incoming wave angle [deg].
        xs: float, optional
            If provided, the current position of the shoreline [m].
        """

        # KA: I added this function because I need to be able to modify the
        # initialization parameters, of which the variables below are dependent

        self._RNG = np.random.default_rng(seed=1973)

        ###############################################################################
        # Dependent variables
        ###############################################################################

        self._u_e_star = self._u_e / np.sqrt(
            self._g * self._a0
        )  # equilibrium inlet velocity (non-dimensional)
        self._Vd_max = self._w_b_crit * self._h_b_crit  # max deficit volume m3/m
        w_s = (
            self._R
            * self._g
            * self._grain_size ** 2
            / (
                (18 * 1e-6)
                + np.sqrt(0.75 * self._R * self._g * (self._grain_size ** 3))
            )
        )  # [m/s] church ferguson 2004
        phi = (
            16 * self._e_s * self._c_s / (15 * np.pi * self._R * self._g)
        )  # phi from aleja/ashton and trueba/ashton
        self._z0 = (
            2 * self._wave_height / 0.78
        )  # minimum depth of integration (very simple approximation of breaking wave depth based on offshore wave height)
        self._d_sf = (
            8.9 * self._wave_height
        )  # 0.018*wave_height*wave_period*sqrt(g/(R*grain_size)); #depth shoreface m #Hallermeier (1983) or  houston (1995)
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
        )
        self._s_sf_eq = (
            3
            * w_s
            / 4
            / np.sqrt(self._d_sf * self._g)
            * (5 + 3 * self._wave_period ** 2 * self._g / 4 / (np.pi ** 2) / self._d_sf)
        )
        # equilibrium shoreface slope
        # self._wave_cdf = np.cumsum(
        #     4
        #     * np.r_[
        #         (
        #             self._wave_asym
        #             * self._wave_high
        #             * np.ones((int(self._wave_climl / 4), 1))
        #         ),
        #         (
        #             self._wave_asym
        #             * (1 - self._wave_high)
        #             * np.ones((int(self._wave_climl / 4), 1))
        #         ),
        #         (
        #             (1 - self._wave_asym)
        #             * (1 - self._wave_high)
        #             * np.ones((int(self._wave_climl / 4), 1))
        #         ),
        #         (
        #             (1 - self._wave_asym)
        #             * self._wave_high
        #             * np.ones((int(self._wave_climl / 4), 1))
        #         ),
        #     ]
        #     / self._wave_climl
        # )
        wave_pdf = np.concatenate(
            4
            * np.r_[
                (
                    self._wave_asym
                    * self._wave_high
                    * np.ones((int(self._wave_climl / 4), 1))
                ),
                (
                    self._wave_asym
                    * (1 - self._wave_high)
                    * np.ones((int(self._wave_climl / 4), 1))
                ),
                (
                    (1 - self._wave_asym)
                    * (1 - self._wave_high)
                    * np.ones((int(self._wave_climl / 4), 1))
                ),
                (
                    (1 - self._wave_asym)
                    * self._wave_high
                    * np.ones((int(self._wave_climl / 4), 1))
                ),
            ]
            / self._wave_climl
        )
        self._coast_qs = (
            self._wave_height ** 2.4
            * (self._wave_period ** 0.2)
            * 3600
            * 365
            * 24
            * self._k
            * (np.cos(self._angle_array) ** 1.2)
            * np.sin(self._angle_array)
        )  # [m3/yr]
        # KA: note first and last points are different here from Matlab version
        self._coast_diff = np.convolve(
            wave_pdf,
            -(
                self._k
                / (self._h_b_crit + self._d_sf)
                * self._wave_height ** 2.4
                * self._wave_period ** 0.2
            )
            * 365
            * 24
            * 3600
            * (np.cos(self._angle_array) ** 0.2)
            * (1.2 * np.sin(self._angle_array) ** 2 - np.cos(self._angle_array) ** 2),
            mode="same",
        )
        # [m2/yr]

        # timestepping implicit diffusion equation (KA: -1 for python indexing)
        self._di = (
            np.r_[
                self._ny,
                np.arange(2, self._ny + 1),
                np.arange(1, self._ny + 1),
                np.arange(1, self._ny),
                1,
            ]
            - 1
        )
        self._dj = (
            np.r_[
                1,
                np.arange(1, self._ny),
                np.arange(1, self._ny + 1),
                np.arange(2, self._ny + 1),
                self._ny,
            ]
            - 1
        )

        ###############################################################################
        # Initial conditions
        ###############################################################################

        self._time_index = 1
        self._x_t = (self._z - self._d_sf) / self._s_background + np.zeros(
            self._ny
        )  # position shoreface toe [m]

        # KA - used for seeding, testing discretization
        if self._bseed:
            if xs is None or wave_angle is None:
                raise ValueError("if bseed is True, xs and wave_angle must be provided")
            self._x_s = xs
            # self._wave_angle = wave_angle
        else:
            self._x_s = (
                self._RNG.random(self._ny) + self._d_sf / self._s_sf_eq + self._x_t
            )  # position shoreline [m]
            # self._x_s = np.random.rand(self._ny) + self._d_sf / self._s_sf_eq + self._x_t  # position shoreline [m]

        self._x_b = (
            self._d_sf / self._s_sf_eq + self._w_b_crit + self._x_t
        )  # position back barrier [m]
        self._h_b = 2 + np.zeros(self._ny)  # height barrier [m]
        self._barrier_volume = np.array([])
        self._inlet_idx_close_mat = np.array([])
        self._inlet_idx = (
            []
        )  # KA: originally a matlab cell, here a list that is appended after first time step
        self._inlet_idx_mat = np.array([]).astype(
            float
        )  # KA: we use this variable for NaN operations
        self._inlet_y = np.zeros(self._ny)
        self._y = np.arange(
            100, self._dy * self._ny, self._dy
        )  # alongshore array [KA: just used for plotting]

        # variables used for saving data [KA: changed all self.dt to self._dt]
        self._t = np.arange(
            self._dt, (self._dt * self._nt) + self._dt, self._dt
        )  # time array
        self._Qoverwash = np.float32(np.zeros(int(self._nt)))
        self._Qinlet = np.float32(np.zeros(int(self._nt)))
        self._inlet_age = []
        # KA: changed the saving arrays from matlab version to enable saving every time step in python, e.g., now if I
        # use the default dtsave=1000, the first value in these arrays (i.e., [0]) are the initial conditions and the
        # second value (i.e., [1]) is the first saving index at time_step=1000
        self._inlet_nr = np.uint16(
            np.zeros(np.size(np.arange(0, self._nt, self._dtsave)))
        )
        self._inlet_migr = np.int16(
            np.zeros(np.size(np.arange(0, self._nt, self._dtsave)))
        )
        self._inlet_Qs_in = np.float32(
            np.zeros(np.size(np.arange(0, self._nt, self._dtsave)))
        )
        self._inlet_alpha = np.float32(
            np.zeros(np.size(np.arange(0, self._nt, self._dtsave)))
        )
        self._inlet_beta = np.float32(
            np.zeros(np.size(np.arange(0, self._nt, self._dtsave)))
        )
        self._inlet_delta = np.float32(
            np.zeros(np.size(np.arange(0, self._nt, self._dtsave)))
        )
        self._inlet_ai = np.int32(
            np.zeros(np.size(np.arange(0, self._nt, self._dtsave)))
        )

        # KA - added these back after Eric's rewrite because I needed them for testing
        c_idx = np.uint8(np.zeros((self._ny, 1000)))  # noqa: F841
        bar_strat_x = (  # noqa: F841
            self._x_b[0] + 1000
        )  # cross-shore location where to record stratigraphy. I guess would be better to do it at one instant in time rather than space?
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
        self._s_sf_save[:, 0] = self._s_sf_eq

        # initialize empty arrays for barrier model (added by KA for coupling)
        self._x_t_dt = np.zeros(self._ny)
        self._x_b_dt = np.zeros(self._ny)
        self._x_s_dt = np.zeros(self._ny)
        self._h_b_dt = np.zeros(self._ny)

    @property
    def time_index(self):
        return self._time_index

    @property
    def time_step(self):
        return self._dt

    @property
    def time(self):
        return self.time_index * self.time_step

    # @property
    # def nt(self):
    #     return self._nt

    @property
    def x_t_dt(self):
        return self._x_t_dt

    @property
    def x_s_dt(self):
        return self._x_s_dt

    @property
    def x_b_dt(self):
        return self._x_b_dt

    @property
    def h_b_dt(self):
        return self._h_b_dt

    @property
    def x_t(self):
        return self._x_t

    @property
    def x_s(self):
        return self._x_s

    @property
    def x_b(self):
        return self._x_b

    @property
    def h_b(self):
        return self._h_b

    @property
    def wave_angle(self):
        return self._wave_angle

    @wave_angle.setter
    def wave_angle(self, new_angle):
        if new_angle > 90.0 or new_angle < -90:
            raise ValueError("wave angle must be between -90 and 90 degrees")
        self._wave_angle = wave_angle

    def u(self, a_star, gam, ah_star):
        """new explicit relationship between boundary conditions and inlet area"""
        return np.sqrt(self._g * self._a0) * np.sqrt(
            gam
            / 2.0
            * np.sqrt(a_star)
            * (
                (-gam * np.sqrt(a_star) * ((a_star - ah_star) ** 2))
                + np.sqrt((gam ** 2) * a_star * ((a_star - ah_star) ** 4) + 4)
            )
        )

    def a_star_eq_fun(self, ah_star, gam, u_e_star):
        """pretty function showing how the cross-sectional area varies with different back barrier configurations gamma"""
        return np.real(
            (2 * ah_star) / 3
            + (
                2 ** (2 / 3)
                * cpower(
                    (
                        (
                            18 * ah_star * gam ** 2
                            - 27 * u_e_star ** 4
                            - 2 * ah_star ** 3 * gam ** 2 * u_e_star ** 2
                            + 3
                            * 3 ** (1 / 2)
                            * gam ** 2
                            * u_e_star ** 2
                            * csqrt(
                                -(
                                    4 * ah_star ** 4 * gam ** 4 * u_e_star ** 4
                                    - 4 * ah_star ** 3 * gam ** 2 * u_e_star ** 8
                                    - 8 * ah_star ** 2 * gam ** 4 * u_e_star ** 2
                                    + 36 * ah_star * gam ** 2 * u_e_star ** 6
                                    + 4 * gam ** 4
                                    - 27 * u_e_star ** 10
                                )
                                / (gam ** 4 * u_e_star ** 6)
                            )
                        )
                        / (gam ** 2 * u_e_star ** 2)
                    ),
                    1 / 3,
                )
            )
            / 6
            + (2 ** (1 / 3) * (ah_star ** 2 * u_e_star ** 2 + 3))
            / (
                3
                * u_e_star ** 2
                * cpower(
                    (
                        (
                            18 * ah_star * gam ** 2
                            - 27 * u_e_star ** 4
                            - 2 * ah_star ** 3 * gam ** 2 * u_e_star ** 2
                            + 3
                            * 3 ** (1 / 2)
                            * gam ** 2
                            * u_e_star ** 2
                            * csqrt(
                                -(
                                    4 * ah_star ** 4 * gam ** 4 * u_e_star ** 4
                                    - 4 * ah_star ** 3 * gam ** 2 * u_e_star ** 8
                                    - 8 * ah_star ** 2 * gam ** 4 * u_e_star ** 2
                                    + 36 * ah_star * gam ** 2 * u_e_star ** 6
                                    + 4 * gam ** 4
                                    - 27 * u_e_star ** 10
                                )
                                / (gam ** 4 * u_e_star ** 6)
                            )
                        )
                        / (gam ** 2 * u_e_star ** 2)
                    ),
                    1 / 3,
                )
            )
        )

    def update(self):
        """Update BRIE by a single time step."""
        self._time_index += 1
        # print('time_index=',self._time_index)

        # sea level
        self._z = self._z + (self._dt * self._slr)  # height of sea level
        w = self._x_b - self._x_s  # barrier width
        d_b = np.minimum(
            self._bb_depth * np.ones(np.size(self._x_b)),
            self._z - (self._s_background * self._x_b),
        )  # basin depth
        s_sf = self._d_sf / (self._x_s - self._x_t)  # shoreface slope

        # if the barrier drows, break
        if np.sum(w < -10) > (self._ny / 2) or np.any(w < -1000):
            print("Barrier Drowned - break")
            return
            # break

        if self._barrier_model_on:
            # volume deficit
            Vd_b = np.maximum(0, (self._w_b_crit - w) * (self._h_b + d_b))
            Vd_h = np.maximum(0, (self._h_b_crit - self._h_b) * w)
            Vd = Vd_b + Vd_h

            # overwash fluxes [m^3/m]
            Qow_b = self._dt * self._Qow_max * Vd_b / np.maximum(Vd, self._Vd_max)
            Qow_h = self._dt * self._Qow_max * Vd_h / np.maximum(Vd, self._Vd_max)
            Qow = Qow_b + Qow_h

            # shoreface flux
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
            ) + (2 * self._dt * self._slr / s_sf)
            self._x_s_dt = 2 * Qow / ((2 * self._h_b) + self._d_sf) / (1 - ff) - (
                4
                * Qsf
                * (self._h_b + self._d_sf)
                / (((2 * self._h_b) + self._d_sf) ** 2)
            )
            self._x_b_dt = Qow_b / (self._h_b + d_b)
            self._h_b_dt = (Qow_h / w) - (self._dt * self._slr)

            # how much q overwash w in total [m3/yr]
            self._Qoverwash[self._time_index - 1] = np.sum(self._dy * Qow_b / self._dt)

        elif self._b3d_barrier_model_on:
            # do nothing, x_t_dt, x_s_dt, x_b_dt, and h_b_dt all come from Barrier3d (is there a better way to do this?)
            self._x_t_dt = self._x_t_dt
            self._x_s_dt = self._x_s_dt
            self._x_b_dt = self._x_b_dt
            self._h_b_dt = self._h_b_dt

        else:
            self._x_t_dt = np.zeros(self._ny)
            self._x_s_dt = np.zeros(self._ny)
            self._x_b_dt = np.zeros(self._ny)
            self._h_b_dt = np.zeros(self._ny)

        if (
            self._ast_model_on
        ):  # only alongshore transport calculation to estimate flux into inlets

            # simple conv approach - KA: example of first row appended to the end
            theta = (
                180
                * (
                    np.arctan2(
                        (self._x_s[np.r_[1 : len(self._x_s), 0]] - self._x_s), self._dy
                    )
                )
                / np.pi
            )

            wave_ang = self._wave_angle
            # wave direction
            # if self._bseed:
            #     wave_ang = self._wave_angle[self._time_index - 1]
            # else:
            #     # wave_ang = np.nonzero(self._wave_cdf > np.random.rand())[0][
            #     wave_ang = np.nonzero(self._wave_cdf > self._RNG.random())[0][
            #         0
            #     ]  # just get the first nonzero element

            # sed transport this timestep (KA: NOTE, -1 indexing is for Python)
            Qs = (
                self._dt
                * self._coast_qs[
                    np.minimum(
                        self._wave_climl,
                        np.maximum(
                            1,
                            np.round(
                                self._wave_climl
                                - wave_ang
                                - (self._wave_climl / 180 * theta)
                                + 1
                            ),
                        ),
                    ).astype(int)
                    - 1
                ]
            ).astype(float)

        if self._inlet_model_on:

            # array for changes to back barrier due to flood tidal deltas
            x_b_fld_dt = np.zeros(int(self._ny))

            # KA, note this was originally empty
            # barrier volume is barrier width times height + estimated inlet depth
            self._barrier_volume = (
                w * (self._h_b + 2) * np.sign(np.minimum(w, self._h_b))
            )

            # KA: added if statement here because error thrown for empty list
            if (
                np.size(self._inlet_idx) != 0
            ):  # KA: inlet_idx is a list here with arrays of different size (from previous time loop)
                self._barrier_volume[np.hstack(self._inlet_idx)] = np.inf

                # add drowned barrier to list of inlets
                self._inlet_idx.append(np.nonzero(self._barrier_volume < 0)[0])

            # storm for new inlet every 10 year
            if (
                np.mod(self._t[self._time_index - 1], 10) < (self._dt / 2)
                and np.size(self._inlet_idx) < self._inlet_max
            ):

                # potential basin length
                if np.size(self._inlet_idx) == 0:
                    basin_length = self._Jmin + np.zeros(int(self._ny)).astype(float)
                else:
                    # KA: there might be a more sophisticated way to replicate
                    # bsxfun (maybe utilizing the inherent broadcasting
                    # capabilities in python), but alas ... had to divide into two operations
                    self._inlet_idx_mat = np.hstack(self._inlet_idx)
                    # KA: this one is confusing to debug if comparing to the same
                    # inlet indices in Matlab because of the zero indexing in Python,
                    # but note that it appears that Lmin/2 is applied in each direction
                    basin_length = np.ravel(
                        (
                            np.array(([-self._ny, 0, self._ny]))
                            + np.reshape(
                                self._inlet_idx_mat + 1,
                                (np.size(self._inlet_idx_mat), 1),
                            )
                        ).T
                    )
                    basin_length = np.amin(
                        np.minimum(
                            self._Jmin,
                            2
                            * self._dy
                            * np.abs(
                                np.arange(1, self._ny + 1, 1)
                                - np.reshape(basin_length, (np.size(basin_length), 1))
                            ),
                        ),
                        axis=0,
                    )

                # basin width is simpler (cross-barrier basin width [m])
                self._basin_width = np.maximum(
                    0, self._z / self._s_background - self._x_b
                )

                # find new inlets only if its far enough away from existing inlets
                # KA: i.e., find the instances of Jmin
                idx = np.nonzero(basin_length > (self._Jmin - 1))[0]

                # KA: for Python, we need to check if there is an instance of Jmin,
                # else an error will be thrown with argmin. If the number of inlets
                # has been saturated, then we keep new_inlet empty from previous loop
                if np.size(idx) != 0:
                    self._new_inlet = np.argmin(
                        self._barrier_volume[idx]
                    )  # KA: find the instance of Jmin at the narrowest point
                    self._new_inlet = idx[self._new_inlet]

                # add new breach to list of inlets
                self._inlet_idx.append(
                    np.array(self._new_inlet)
                )  # KA: not sure if I need the np.array here

            # get rid of duplicates and neighbours
            if np.size(self._inlet_idx) != 0:
                # KA: inlet_idx_mat is just inlet_idx concatenated into a single
                # array and made a float so we can use NaNs
                self._inlet_idx_mat = np.hstack(self._inlet_idx).astype(float)
                inlet_all_idx = np.sort(self._inlet_idx_mat)
                inlet_all_idx_idx = np.argsort(self._inlet_idx_mat)

                # don't try to understand this line (message from Jaap)
                # KA: replaced empty "cell" contents with NaNs
                self._inlet_idx_mat[
                    inlet_all_idx_idx[
                        np.less_equal(
                            np.diff(np.r_[inlet_all_idx[-1] - self._ny, inlet_all_idx]),
                            1,
                        )
                    ]
                ] = np.nan
                self._inlet_idx_mat = self._inlet_idx_mat[
                    ~np.isnan(self._inlet_idx_mat)
                ]
                # KA: here, inlet_idx is reduced to just the first index (still a list)
                self._inlet_idx = self._inlet_idx_mat.astype(int).tolist()

            # do "fluid mechanics" of inlets (KA: I see no need for the second
            # "if" statement, but leave as is )
            if np.size(self._inlet_idx) != 0:
                # sort inlets (first index only) and find respective tidal prisms
                inlet_all_idx = np.sort(self._inlet_idx)
                inlet_all_idx_idx = np.argsort(self._inlet_idx)
                inlet_dist = np.diff(
                    np.r_[
                        inlet_all_idx[-1] - self._ny,
                        inlet_all_idx,
                        inlet_all_idx[0] + self._ny,
                    ]
                )  # KA: distance between inlets
                basin_length = np.minimum(
                    self._Jmin,
                    (
                        self._dy
                        * 0.5
                        * (inlet_dist[0:-1] + inlet_dist[1 : len(inlet_dist)])
                    ),
                )

                # see swart zimmerman
                ah_star = (
                    self._omega0 * w[self._inlet_idx] / np.sqrt(self._g * self._a0)
                )
                c_d = self._g * self._man_n ** 2 / (d_b[self._inlet_idx] ** (1 / 3))
                gam = np.maximum(
                    1e-3,
                    self._inlet_asp
                    * (
                        (self._omega0 ** 2)
                        * (1 - self._marsh_cover) ** 2
                        * (basin_length[inlet_all_idx_idx] ** 2)
                        * (self._basin_width[self._inlet_idx] ** 2)
                        * self._a0
                        / self._g
                    )
                    ** (1 / 4)
                    / ((8 / 3 / np.pi) * c_d * w[self._inlet_idx]),
                )
                a_star_eq = self.a_star_eq_fun(ah_star, gam, self._u_e_star)
                u_eq = np.real(self.u(a_star_eq, gam, ah_star))
                ai_eq = (
                    self._omega0
                    * (1 - self._marsh_cover)
                    * basin_length[inlet_all_idx_idx]
                    * self._basin_width[self._inlet_idx]
                    * np.sqrt(self._a0 / self._g)
                ) * a_star_eq  # KA: does it matter that this was last defined during the Tstorm year?

                # keep inlet open if velocity is at equilibrium (Escoffier); add
                # margin of 0.05 m/s for rounding errors etc
                inlet_close = np.logical_and(
                    np.logical_or(np.less(u_eq, (self._u_e - 0.05)), np.isnan(u_eq)),
                    np.greater(w[self._inlet_idx], 0),
                )

                # we don't have to think about this one every again!
                self._inlet_idx_mat[
                    inlet_close
                ] = np.nan  # KA: use inlet_idx_mat b/c float
                self._inlet_idx_close_mat = np.argwhere(
                    np.isnan(self._inlet_idx_mat)
                )  # KA: get index
                self._inlet_idx_mat = self._inlet_idx_mat[
                    ~np.isnan(self._inlet_idx_mat)
                ]
                # KA: again here, inlet_idx is just the first index (still a list), and not sorted
                self._inlet_idx = self._inlet_idx_mat.astype(int).tolist()
                ai_eq[inlet_close] = np.nan
                ai_eq = ai_eq[~np.isnan(ai_eq)]

                wi_eq = np.sqrt(ai_eq) / self._inlet_asp  # calculate width and depths
                di_eq = ai_eq / wi_eq
                wi_cell = np.ceil(wi_eq / self._dy).astype(
                    int
                )  # get cell widths per inlet

            # KA: python object arrays to "mimic" Matlab cells for inlet tracking
            # in retrospect, probably didn't need objects. Empty list would have been fine.
            inlet_nex = np.empty(np.size(self._inlet_idx), dtype=object)
            inlet_prv = np.empty(np.size(self._inlet_idx), dtype=object)

            # preallocate arrays for inlet migration and fractions based on I
            migr_up, delta, beta, beta_r, alpha, alpha_r, delta_r, Qs_in = [
                np.zeros(np.size(self._inlet_idx)).astype(float) for _ in range(8)
            ]

            # inlet morphodynamics per inlet (KA: again, j-1 here for python)
            for j in np.arange(1, np.size(self._inlet_idx) + 1):

                # breach sediment is added to the flood-tidal delta
                if (
                    self._new_inlet.size > 0
                    and self._inlet_idx[j - 1] == self._new_inlet
                ):  # KA: for python, need to check that the array isn't empty
                    # KA: here, Jaap allows the indexing to wrap such that a new
                    # inlet formed at the end of the model domain can deposit sediment
                    # at the start of the model domain; does this wrapping for all
                    # inlet dynamics (see other np.mods throughout code)
                    new_inlet_idx = np.mod(
                        self._new_inlet + np.r_[1 : (wi_cell[j - 1] + 1)] - 1, self._ny
                    )
                    x_b_fld_dt[new_inlet_idx] = x_b_fld_dt[new_inlet_idx] + (
                        (self._h_b[self._new_inlet] + di_eq[j - 1]) * w[self._new_inlet]
                    ) / (d_b[self._new_inlet])

                    self._Qinlet[self._time_index - 1] = self._Qinlet[
                        self._time_index - 1
                    ] + (
                        (self._h_b[self._new_inlet] + d_b[self._new_inlet])
                        * w[self._new_inlet]
                        * wi_cell[j - 1]
                        * self._dy
                    )

                # alongshore flux brought into inlet
                Qs_in[j - 1] = Qs[self._inlet_idx[j - 1]]

                # find cells of inlet, updrift barrier, and downdrift barrier
                # KA: here, inlet_idx becomes a list of arrays, and again,
                # inlets wrap around the edges
                self._inlet_idx[j - 1] = np.mod(
                    self._inlet_idx[j - 1] + np.r_[1 : (wi_cell[j - 1] + 1)] - 1,
                    self._ny,
                ).astype(int)
                inlet_nex[j - 1] = np.mod(self._inlet_idx[j - 1][-1] + 1, self._ny)
                inlet_prv[j - 1] = np.mod(self._inlet_idx[j - 1][0] - 1, self._ny)

                # find momentum balance of inlet to determine sediment
                # distribution fractions
                Mt = self._rho_w * self._u_e * self._u_e * ai_eq[j - 1]
                Mw = self._rho_w / 16 * self._g * self._wave_height ** 2 * wi_eq[j - 1]
                I = Mt / Mw * wi_eq[j - 1] / w[self._inlet_idx[j - 1][0]]
                self._h_b[self._inlet_idx[j - 1]] = 0

                # constrain to not widen
                Ab_prv = w[inlet_prv[j - 1]] * (
                    self._h_b[self._inlet_idx[j - 1][0]] + di_eq[j - 1]
                )
                Ab_nex = w[inlet_nex[j - 1]] * (
                    self._h_b[inlet_nex[j - 1]] + di_eq[j - 1]
                )

                # do fld delta eq volume
                Vfld = (
                    (
                        self._x_b[self._inlet_idx[j - 1][0]]
                        - self._x_s[self._inlet_idx[j - 1][0]]
                        + self._w_b_crit
                    )
                    * wi_eq[j - 1]
                    * d_b[self._inlet_idx[j - 1][0]]
                )
                Vfld_max = 1e4 * (self._u_e * ai_eq[j - 1] / 2 / self._omega0) ** 0.37

                # add fix to limit unrealistic flood-tidal delta size (based on
                # johnson flood-tidal delta of florida 2006)
                if Vfld > Vfld_max:
                    I = 0.1

                # calculate fractions based on I (KA: added self here b/c otherwise it produced an error)
                # version 1: original BRIE upload (shoreline not stable)
                # delta[j - 1] = inlet_fraction(self, 0, 1, 3, -3, I)
                # beta[j - 1] = inlet_fraction(self, 0, 1, 10, 3, I)
                # beta_r[j - 1] = inlet_fraction(self, 0, 1, 0.9, -3, I)

                # version 2: from Neinhuis and Ashton 2016 (updated May 27, 2020)
                # NOTE, Jaap only changes beta_r in the Matlab version, so we will do the same here
                # delta[j - 1] = inlet_fraction(self, 0.05, 0.95, 3, -3, I)
                # beta[j - 1] = inlet_fraction(self, 0, 0.9, 10, 3, I)
                # beta_r[j - 1] = inlet_fraction(self, 0, 0.9, 0.9, -3, I)

                delta[j - 1] = inlet_fraction(self, 0, 1, 3, -3, I)
                beta[j - 1] = inlet_fraction(self, 0, 1, 10, 3, I)
                beta_r[j - 1] = inlet_fraction(self, 0, 0.9, 0.9, -3, I)

                #            #{
                #            humans affect inlets?
                #            delta(j) = 0;
                #            beta(j) = 1;
                #            beta_r(j) = 0;
                #              #}

                alpha[j - 1] = 1 - beta[j - 1] - delta[j - 1]
                alpha_r[j - 1] = alpha[j - 1] * 0.6

                if Vfld > Vfld_max:
                    delta_r[j - 1] = 0
                else:
                    delta_r[j - 1] = (
                        (Ab_nex * alpha[j - 1]) - (Ab_prv * beta_r[j - 1])
                    ) / Ab_prv

                # use fractions to physically move inlets and fld-tidal detlas

                # update fld delta, deposit sediment at 0 water depth
                fld_delta = np.abs(Qs_in[j - 1]) * (
                    delta[j - 1] + delta_r[j - 1]
                )  # KA: this seems low for my QC scenario, maybe come back to this
                # remove sediment from the shoreface
                inlet_sink = np.abs(Qs_in[j - 1]) * (1 - beta[j - 1] - beta_r[j - 1])
                # spread fld tidal delta along one more cell alongshore in both directions
                temp_idx = np.r_[
                    inlet_prv[j - 1], self._inlet_idx[j - 1], inlet_nex[j - 1]
                ]

                x_b_fld_dt[temp_idx] = x_b_fld_dt[temp_idx] + fld_delta / (
                    np.size(temp_idx) * self._dy
                ) / (self._h_b[temp_idx] + d_b[temp_idx])

                # migrate inlet indices (in m/dt)
                migr_up[j - 1] = Qs_in[j - 1] * (alpha_r[j - 1] + alpha[j - 1]) / Ab_prv
                #                migr_dw = (
                #                    Qs_in[j - 1]
                #                    * (alpha_r[j - 1] + beta_r[j - 1] + delta_r[j - 1])
                #                    / Ab_nex
                #                )

                # calculate where in the grid cell the inlet is, and add the
                # fractional migration to it
                self._inlet_y[self._inlet_idx[j - 1][0]] = (
                    self._inlet_y[self._inlet_idx[j - 1][0]] + migr_up[j - 1] / self._dy
                )

                # how far are the inlets in their gridcell?
                # (or is inlet_y>1 or <0 and should the inlet hop one grid cell?)
                migr_int = np.floor(
                    self._inlet_y[self._inlet_idx[j - 1][0]]
                )  # KA: unsure about these too
                migr_res = np.mod(self._inlet_y[self._inlet_idx[j - 1][0]], 1)

                # reset old grid cell
                self._inlet_y[self._inlet_idx[j - 1][0]] = 0

                # move inlet in gridcell
                self._inlet_idx[j - 1] = np.mod(
                    self._inlet_idx[j - 1] + migr_int, self._ny
                ).astype(int)

                self._inlet_y[self._inlet_idx[j - 1][0]] = migr_res

                # how much q flood tidal delta in total
                self._Qinlet[self._time_index - 1] = (
                    self._Qinlet[self._time_index - 1] + inlet_sink
                )  # m3 per time step

                # add inlet sink to shoreline change (updated May 27, 2020 so that shoreline change from inlet sink
                # now spread out along width of inlet +1 cell in both directions)
                # self._x_s_dt[inlet_nex[j - 1]] = (
                #         self._x_s_dt[inlet_nex[j - 1]]
                #         + inlet_sink / (self._h_b[inlet_nex[j - 1]] + self._d_sf) / self._dy
                # )
                self._x_s_dt[temp_idx] = (
                    self._x_s_dt[temp_idx]
                    + inlet_sink
                    / (self._h_b[temp_idx] + self._d_sf)
                    / len(temp_idx)
                    / self._dy
                )

                # inlet age
                # fancy lightweight way to keep track of where inlets are in the model
                # KA: note that this differs from matlab version, here we do this all
                # in the for loop
                self._inlet_age.append(
                    [self._time_index, self._inlet_idx[j - 1][0].astype("int32")]
                )

            # reset arrays
            self._new_inlet = np.array([])

            # inlet statistics
            if (
                np.mod(self._time_index, self._dtsave) == 0
            ):  # KA: modified this from matlab version so that I can save every time step in python
                # skip first time step (initial condition)
                self._inlet_nr[
                    np.fix(self._time_index / self._dtsave).astype(int) - 1
                ] = len(
                    self._inlet_idx
                )  # number of inlets
                self._inlet_migr[
                    np.fix(self._time_index / self._dtsave).astype(int) - 1
                ] = np.mean(migr_up / self._dt)

                if np.size(self._inlet_idx) != 0:
                    self._inlet_Qs_in[
                        np.fix(self._time_index / self._dtsave).astype(int) - 1
                    ] = np.mean(Qs_in)
                    self._inlet_alpha[
                        np.fix(self._time_index / self._dtsave).astype(int) - 1
                    ] = np.mean(alpha)
                    self._inlet_beta[
                        np.fix(self._time_index / self._dtsave).astype(int) - 1
                    ] = np.mean(beta)
                    self._inlet_delta[
                        np.fix(self._time_index / self._dtsave).astype(int) - 1
                    ] = np.mean(delta)
                    self._inlet_ai[
                        np.fix(self._time_index / self._dtsave).astype(int) - 1
                    ] = np.mean(ai_eq)

        else:  # inlet model not on
            # Qs_in = 0
            # delta = 0
            # delta_r = 0
            # inlet_sink = 0
            x_b_fld_dt = 0

        # do implicit thing (updated on May 27, 2020 to force shoreline diffusivity to be greater than zero)
        if self._ast_model_on:
            r_ipl = np.maximum(
                0,
                (
                    self._coast_diff[
                        np.maximum(
                            1,
                            np.minimum(
                                self._wave_climl, np.round(90 - theta).astype(int)
                            ),
                        )
                    ]
                    * self._dt
                    / 2
                    / self._dy ** 2
                ),
            )

            dv = np.r_[-r_ipl[-1], -r_ipl[1:], 1 + 2 * r_ipl, -r_ipl[0:-1], -r_ipl[0]]
            A = csr_matrix((dv, (self._di, self._dj)))
            # A = sps.sparse.csr_matrix((dv, (self._di, self._dj))).toarray()  # KA: spot checked, but probably worth a closer look

            RHS = (
                self._x_s
                + r_ipl
                * (
                    self._x_s[np.r_[1 : self._ny, 0]]
                    - 2 * self._x_s
                    + self._x_s[np.r_[self._ny - 1, 0 : self._ny - 1]]
                )
                + self._x_s_dt
            )

            self._x_s = spsolve(
                A, RHS
            )  # KA: will want to check this as a diff - accidentally overwrote while checking
        else:
            self._x_s = self._x_s + self._x_s_dt

        # how are the other moving boundaries changing?
        self._x_t = self._x_t + self._x_t_dt
        self._x_b = self._x_b + self._x_b_dt + x_b_fld_dt
        self._h_b = self._h_b + self._h_b_dt

        # save subset of BRIE variables (KA: I changed this from mod = 1 to mod = 0 to allow for saving every 1 timestep)
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

        # self.__dict__  # to look at attributes
        # del(self._t)
        del self._inlet_y
        del self._inlet_idx
        del self._inlet_idx_mat
        del self._inlet_idx_close_mat
        del self._barrier_volume
        del self._h_b
        del self._x_b
        del self._x_s
        del self._x_t
        del self._di
        del self._dj
        del self._coast_diff
        del self._coast_qs

        if self._inlet_model_on:
            self._Qinlet = self._Qinlet / self._dt  # put into m3/yr
            # self._Qinlet_norm = (self._Qinlet / self._dy)  # put into m3/m/yr
        else:
            del self._inlet_Qs_in
            del self._inlet_migr
            del self._inlet_nr
            del self._inlet_age
            del self._Qinlet
            del self._inlet_ai
            del self._inlet_delta
            del self._inlet_beta
            del self._inlet_alpha
