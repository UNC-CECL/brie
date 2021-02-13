"""Alongshore sediment transport

This module provides function for calculating alongshore sediment transport
based on Nienhuis et al., 2015 [1]_ .

References
----------

.. [1] Jaap H. Nienhuis, Andrew D. Ashton, Liviu Giosan; What makes a delta wave-dominated?. Geology ; 43 (6): 511–514. doi: https://doi.org/10.1130/G36518.1
"""
import numpy as np
import scipy.constants
import scipy.sparse

SECONDS_PER_YEAR = 3600.0 * 24.0 * 365.0


def calc_alongshore_transport_k(
    gravity=scipy.constants.g,
    n=1.0,
    rho_water=1050.0,
    gamma_b=0.78,
):
    r"""Calculate alongshore transport diffusion coefficient.

    The diffusion coefficient is calculated from Nienhuis, Ashton, Giosan, 2015 [1]_ .
    Note that the Ashton, 2006 value for *k* is incorrect.

    Parameters
    ----------
    gravity : float, optional
        Acceleration due to gravity [m/s^2].
    n : float, optional
        Ratio of group velocity to phase velocity of the breaking waves
        (1 in shallow water).
    rho_water: float, optional
        Density of water [kg / m^3].
    gamma_b: float, optional
        Ratio of breaking wave height and water depth.

    Returns
    -------
    float
        Empirical constant for alongshore sediment transport.

    Notes
    -----

    The sediment transport constant, :math:`K_1`, is calculated as follows,

    .. math::

        K_1 = 5.3 \cdot 10^{-6} K \left( \frac{1}{2n} \right)^{6 \over 5} \left( \frac{\sqrt{g \gamma_b}} {2 \pi} \right)^{1 \over 5}

    where:

    .. math::

        K = 0.46 \rho g^{3 \over 2}

    """
    return (
        5.3e-6
        # * 0.46  # I'm not sure about this factor
        * rho_water
        * gravity ** 1.5
        * (1 / (2 * n)) ** 1.2
        * (np.sqrt(gravity * gamma_b) / (2 * np.pi)) ** 0.2
    )


def calc_shoreline_angles(y, spacing=1.0, out=None):
    r"""Calculate shoreline angles.

    Given a series of coastline positions, `y`, with equal spacing
    of points, calculate coastline angles with the *x*-axis. Angles
    at first and last points are calculated using wrap-around
    boundaries.

    Parameters
    ----------
    y : array of float
        Y-positions of the shoreline [m].
    spacing : float
        Spacing between shoreline segments [m].
    out : ndarray or None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned.

    Returns
    -------
    angles : array of float
        Angle of shoreline [rads].

    Examples
    --------
    >>> import numpy as np
    >>> from brie.alongshore_transporter import calc_shoreline_angles

    >>> y = [0.0, 0.0, 0.0, 0.0, 0.0]
    >>> calc_shoreline_angles(y)
    array([0., 0., 0., 0., 0.])

    Angles are measures with respect to the x-axis.

    >>> y = [0.0, 0.0, 1.0, 0.0, 0.0]
    >>> np.rad2deg(calc_shoreline_angles(y))
    array([  0.,  45., -45.,   0.,   0.])

    Angles at the ends are calculated as wrap-around boundaries.

    >>> y = [1.0, 0.0, 0.0, 0.0, 0.0]
    >>> np.rad2deg(calc_shoreline_angles(y))
    array([-45.,   0.,   0.,   0.,  45.])
    """
    return np.arctan2(np.diff(y, append=y[0]), spacing, out=out)


def calc_coast_qs(wave_angle, wave_height=1.0, wave_period=10.0):
    r"""Calculate coastal alongshore sediment transport for a given incoming wave angle.

    Parameters
    ----------
    wave_angle: float or array of float
        Incoming wave angle relative to local shoreline normals [rad]. That is, a
        value of 0 means approaching waves are normal to the coast, negative
        values means waves approaching from the right, and positive from
        the left.
    wave_height: float, optional
        Height of incoming waves [m].
    wave_period: float, optional
        Period of incoming waves [s].

    Returns
    -------
    float or array of float
        Coastal qs [m3 / yr]

    Notes
    -----

    Alongshore sediment transport is computed using the CERC or Komar (Komar, 1998 [2]_ ) formula, reformulated into deep-water wave properties (Ashton and Murray, 2006 [3]_ ) by back-refracting the waves over shore-parallel contours, which yields:

    .. math::

        Q_s = K_1 \cdot H_s^{12/5} T^{1/5} \cos^{6/5}\left( \Delta \theta \right) \sin \left(\Delta \theta\right)

    where :math:`H_s` is the offshore deep-water significant wave height (in meters), :math:`T` is the wave period (in seconds), and :math:`\Delta \theta` is the deep-water wave approach angle relative to the local shoreline orientation (rads).

    References
    ----------
    .. [2] Komar P.D., 1998, Beach processes and sedimentation: Upper Saddle River, New Jersey, Prentice Hall , 544 p.

    .. [3] Ashton A.D. Murray A.B., 2006, High-angle wave instability and emergent shoreline shapes: 1. Modeling of sand waves, flying spits, and capes: Journal of Geophysical Research , v. 111, F04011, doi:10.1029/2005JF000422.
    """
    return (
        wave_height ** 2.4
        * (wave_period ** 0.2)
        * SECONDS_PER_YEAR
        * AlongshoreTransporter.K
        * (np.cos(wave_angle) ** 1.2)
        * np.sin(wave_angle)
    )  # [m3/yr]


def calc_alongshore_transport(
    wave_angle, shoreline_angle=0.0, wave_height=1.0, wave_period=10.0
):
    """Calculate alongshore transport along a coastline.

    Parameters
    ----------
    wave_angle: float
        Incoming wave angle as measured counter-clockwise from the
        positive x-axis [rads].
    shoreline_angle: float or array of float, optional
        Angle of shoreline with respect to the positive x-axis [rads].
    wave_height: float, optional
        Incoming wave height [m].
    wave_period: float, optional
        Incoming wave period [s].

    Returns
    -------
    float or array of float
        Alongshore transport along the shoreline.
    """
    wave_angle_wrt_shoreline = np.clip(
        # np.pi / 2.0 + shoreline_angle - wave_angle,
        wave_angle - shoreline_angle,
        a_min=-np.pi / 2.0,
        a_max=np.pi / 2.0,
    )

    return calc_coast_qs(
        wave_angle_wrt_shoreline, wave_height=wave_height, wave_period=wave_period
    )


def calc_coast_diff(
    wave_pdf, wave_angle, wave_height=1.0, wave_period=10.0, h_b_crit=2.0
):
    r"""Calculate sediment diffusion along a coastline.

    .. math::

        Q_{s,net} \left( \theta \right) = E \left( \phi_0 \right) * Q_s \left( \phi_0 - \theta \right)

    Parameters
    ----------
    wave_pdf: func
        Probability density function of incoming waves defined for wave
        angles from -pi / 2 to pi / 2.
    wave_angle: float
        Incoming wave angle with respect to coast normal [rad].
    wave_height: float, optional
        Height of incoming waves [m].
    wave_period: float, optional
        Period of incoming waves [s].
    """
    all_angles, step = np.linspace(-np.pi / 2.0, np.pi / 2.0, 181, retstep=True)

    d_sf = 8.9 * wave_height

    # delta_angles = np.clip(
    #     wave_angle - shoreline_angles, a_min=-np.pi / 2.0, a_max=np.pi / 2.0
    # )

    y = np.convolve(
        wave_pdf(all_angles) * step,
        -(
            AlongshoreTransporter.K
            / (h_b_crit + d_sf)
            * wave_height ** 2.4
            * wave_period ** 0.2
        )
        * SECONDS_PER_YEAR
        # * (np.cos(delta_angles) ** 0.2)
        # * (1.2 * np.sin(delta_angles) ** 2 - np.cos(delta_angles) ** 2),
        * (np.cos(all_angles) ** 0.2)
        * (1.2 * np.sin(all_angles) ** 2 - np.cos(all_angles) ** 2),
        mode="same",
    )

    # return np.interp(shoreline_angles, all_angles, y) * np.sign(-wave_angle)
    return np.interp(-wave_angle, all_angles, y)  #  * np.sign(-wave_angle)


def _build_tridiagonal_matrix(diagonal, lower=None, upper=None):
    """Build a tridiagonal matrix with wrap-around boundaries.

    Parameters
    ----------
    values_at_node: array of float
        Values to place along the diagonals.

    Examples
    --------
    >>> from brie.alongshore_transporter import _build_tridiagonal_matrix
    >>> _build_tridiagonal_matrix([1, 2, 3, 4]).toarray()
    array([[1, 1, 0, 1],
           [2, 2, 2, 0],
           [0, 3, 3, 3],
           [4, 0, 4, 4]])

    >>> _build_tridiagonal_matrix(
    ...     [1, 2, 3, 4], lower=[11, 12, 13, 14], upper=[21, 22, 23, 24]
    ... ).toarray()
    array([[ 1, 21,  0, 11],
           [12,  2, 22,  0],
           [ 0, 13,  3, 23],
           [24,  0, 14,  4]])
    """
    if lower is None:
        lower = diagonal
    if upper is None:
        upper = diagonal
    n_rows = n_cols = len(diagonal)

    mat = scipy.sparse.spdiags(
        [np.r_[lower[1:], 0], diagonal, np.r_[0, upper[:-1]]],
        [-1, 0, 1],
        n_rows,
        n_cols,
    ).tolil()

    mat[0, -1] = lower[0]
    mat[-1, 0] = upper[-1]

    return mat


def _build_matrix(
    shoreline_x, wave_distribution, dy=1.0, wave_height=1.0, wave_period=10.0
):
    dt = 1.0

    shoreline_angles = calc_shoreline_angles(shoreline_x, spacing=dy)

    r_ipl = np.clip(
        calc_coast_diff(
            wave_distribution.pdf,
            # np.pi / 2.0 - shoreline_angles, # Use shoreline angles???
            -shoreline_angles,  # Use shoreline angles???
            # angles, # Use shoreline angles???
            wave_height=wave_height,
            wave_period=wave_period,
        )
        * dt
        / (2.0 * dy ** 2),
        a_min=0.0,
        a_max=None,
    )

    mat = _build_tridiagonal_matrix(1.0 + 2.0 * r_ipl, lower=-r_ipl, upper=-r_ipl)

    rhs = (
        shoreline_x
        + r_ipl
        * np.diff(
            shoreline_x,
            n=2,
            prepend=shoreline_x[-1:],
            append=shoreline_x[:1],
        )
        # + self._x_s_dt
    )

    return mat.tocsc(), rhs, r_ipl


class AlongshoreTransporter:

    """Transport sediment along a coast.

    Examples
    --------
    >>> from brie.alongshore_transporter import AlongshoreTransporter
    >>> transporter = AlongshoreTransporter([0.0, 0.0, 1.0, 0.0, 0.0])
    >>> transporter.update()
    """

    K = calc_alongshore_transport_k()

    def __init__(
        self,
        shoreline_x,
        dy=1.0,
        wave_height=1.0,
        wave_period=10.0,
        wave_angle=0.0,
        wave_distribution=None,
    ):
        self._wave_height = wave_height
        self._wave_period = wave_period
        self._wave_angle = wave_angle

        if wave_distribution is None:
            wave_distribution = scipy.stats.uniform(loc=-np.pi / 2.0, scale=np.pi)
        self._wave_distribution = wave_distribution

        self._shoreline_x = np.asarray(shoreline_x, dtype=float)
        self._dy = dy

        self._shoreline_angles = calc_shoreline_angles(
            self._shoreline_x, spacing=self._dy
        )
        self._time = 0.0

        self._q_s = np.empty_like(shoreline_x)

    def _build_matrix(self, dt=1.0):
        shoreline_angles = self._shoreline_angles

        r_ipl = np.clip(
            calc_coast_diff(
                self._wave_distribution.pdf,
                # np.pi / 2.0 - shoreline_angles, # Use shoreline angles???
                -shoreline_angles,  # Use shoreline angles???
                # angles, # Use shoreline angles???
                wave_height=self._wave_height,
                wave_period=self._wave_period,
            )
            * dt
            / (2.0 * self._dy ** 2),
            a_min=0.0,
            a_max=None,
        )

        r_ipl = (
            calc_coast_diff(
                self._wave_distribution.pdf,
                # np.pi / 2.0 - shoreline_angles, # Use shoreline angles???
                -shoreline_angles,  # Use shoreline angles???
                # angles, # Use shoreline angles???
                wave_height=self._wave_height,
                wave_period=self._wave_period,
            )
            * dt
            / (2.0 * self._dy ** 2)
        )

        mat = _build_tridiagonal_matrix(1.0 + 2.0 * r_ipl, lower=-r_ipl, upper=-r_ipl)

        rhs = (
            self._shoreline_x
            + r_ipl
            * np.diff(
                self._shoreline_x,
                n=2,
                prepend=self._shoreline_x[-1:],
                append=self._shoreline_x[:1],
            )
            # + self._x_s_dt
        )

        return mat.tocsc(), rhs

    def update(self, dt=1.0):
        self._time += dt

        self._wave_angle = self._wave_distribution.rvs(size=1)

        self._q_s[:] = (
            calc_alongshore_transport(
                self._wave_angle,
                shoreline_angle=self._shoreline_angles,
                wave_height=self._wave_height,
                wave_period=self._wave_period,
            )
            * dt
        )

        mat, rhs = self._build_matrix(dt=dt)

        self._shoreline_x[:] = scipy.sparse.linalg.spsolve(mat, rhs)
        calc_shoreline_angles(self._shoreline_x, self._dy, out=self._shoreline_angles)

    @property
    def wave_height(self):
        return self._wave_height

    @wave_height.setter
    def wave_height(self, new_val):
        if new_val < 0.0:
            raise ValueError("wave height must be non-negative")
        self._wave_height = new_val

    @property
    def wave_period(self):
        return self._wave_period

    @wave_period.setter
    def wave_period(self, new_val):
        if new_val <= 0.0:
            raise ValueError("wave period must be positive")
        self._wave_period = new_val

    @property
    def wave_angle(self):
        return self._wave_angle
