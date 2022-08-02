"""Inlet dynamics

This module provides functions for opening, closing, and migrating inlets. This includes alongshore sediment transport
into inlets. Formulations described in Nienhuis and Lorenzo-Trueba, 2019 [4]_ and stem from Nienhuis et al., ??? [1]_
Roos et al., 2013, Ashton and Murray, 2006 [2]_.

References
----------

# need to update these
.. [1] Jaap H. Nienhuis, Andrew D. Ashton, Liviu Giosan; What makes a delta wave-dominated?.
    Geology ; 43 (6): 511–514. doi: https://doi.org/10.1130/G36518.1
.. [2] Andrew D. Ashton, A. Brad Murray. High‐angle wave instability and emergent shoreline shapes:
    1. Modeling of sand waves, flying spits, and capes. Journal of Geophysical Research: Earth Surface 111.F4 (2006).
.. [3] P.D. Komar, 1998, Beach processes and sedimentation: Upper Saddle River, New Jersey, Prentice Hall , 544 p.
.. [4] Jaap H. Nienhuis, Jorge Lorenzo Trueba; Simulating barrier island response to sea level rise with the barrier
    island and inlet environment (BRIE) model v1.0 ; Geosci. Model Dev., 12, 4013–4030, 2019; https://doi.org/10.5194/gmd-12-4013-2019


Notes
---------

"""
import numpy as np
import scipy.constants
import scipy.sparse
from numpy.lib.scimath import power as cpower, sqrt as csqrt
from .alongshore_transporter import calc_alongshore_transport_k, calc_shoreline_angles


SECONDS_PER_YEAR = 3600.0 * 24.0 * 365.0
g = scipy.constants.g


def inlet_fraction(a, b, c, d, I):
    """what are the inlet fractions"""
    return a + (b / (1 + c * (I ** d)))

def u(a_star, gam, ah_star, a0):
    """new explicit relationship between boundary conditions and inlet area"""
    return np.sqrt(g * a0) * np.sqrt(
        gam
        / 2.0
        * np.sqrt(a_star)
        * (
            (-gam * np.sqrt(a_star) * ((a_star - ah_star) ** 2))
            + np.sqrt((gam ** 2) * a_star * ((a_star - ah_star) ** 4) + 4)
        )
    )

def a_star_eq_fun(ah_star, gam, u_e_star):
    """pretty function showing how the cross-sectional area varies with different back barrier
    configurations gamma"""
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
        * InletSpinner.K
        * (np.cos(wave_angle) ** 1.2)
        * np.sin(wave_angle)
    )  # [m3/yr]

def calc_inlet_alongshore_transport(
    wave_angle, shoreline_angle=0.0, wave_height=1.0, wave_period=10.0
):
    r"""Calculate alongshore transport along a coastline for a single wave angle. Only used in inlet calculations within
    BRIE.

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

def create_inlet(inlet_idx, ny, dy, barrier_volume, min_inlet_separation=10000):
    r"""Creates a new inlet at the location of minimum barrier volume, but only if inlets are far enough away from
    existing inlets (Roos et al.,2013).

        Parameters
        ----------
        inlet_idx: list of int
            Indices of inlet locations
        ny: int
            Number of alongshore cells
        dy: int
            Length of alongshore cells [m]
        barrier_volume: array
            Barrier width times height + estimated inlet depth
        min_inlet_separation: int
            Minimum separation distance of inlets; from Roos et al., (2013) [m]

        Returns
        -------
        array of integers
            new_inlet: index of the newest inlet
            inlet_idx: indices of all inlets
    """

    new_inlet = []

    # calculate basin length
    if np.size(inlet_idx) == 0:
        basin_length = min_inlet_separation + np.zeros(ny).astype(float)
    else:
        # NOTE TO ERIC: there might be a more sophisticated way to replicate matlab's bsxfun, but alas I had to divide
        # into two operations to get this two work
        inlet_idx_mat = np.hstack(inlet_idx)
        basin_length = np.ravel(
            (
                np.array(([-ny, 0, ny]))
                + np.reshape(
                    inlet_idx_mat + 1,
                    (np.size(inlet_idx_mat), 1),
                )
            ).T
        )  # apply min_inlet_separation/2 in each direction
        basin_length = np.amin(
            np.minimum(
                min_inlet_separation,
                2
                * dy
                * np.abs(
                    np.arange(1, ny + 1, 1)
                    - np.reshape(basin_length, (np.size(basin_length), 1))
                ),
            ),
            axis=0,
        )

    # find new inlets only if its far enough away from existing inlets
    # i.e., check if there is an instance of min_inlet_separation: if the number of inlets has been saturated,
    # then we keep new_inlet empty from previous time loop
    idx = np.nonzero(basin_length > (min_inlet_separation - 1))[0]
    if np.size(idx) != 0:
        new_inlet = np.argmin(
            barrier_volume[idx]
        )  # find the instance of min_inlet_separation at the narrowest point
        new_inlet = idx[new_inlet]

    # add new breach to list of inlets
    inlet_idx.append(np.array(new_inlet))  # KA: not sure if I need the np.array here

    return inlet_idx, new_inlet

def organize_inlet(inlet_idx, ny):
    r"""Gets rid of duplicates and neighbours in the inlet_idx list; inlet_idx is reduced to just the first index of
    the inlet (still a list)

        Parameters
        ----------
        inlet_idx: list of int
            Indices of inlet locations
        ny: int
            Number of alongshore cells

        Returns
        -------
        array of integers
            inlet_idx: indices of all inlets
    """

    # inlet_idx_mat is just inlet_idx concatenated into a single array and made a float so we can use NaNs
    inlet_idx_mat = np.hstack(inlet_idx).astype(float)
    inlet_all_idx = np.sort(inlet_idx_mat)
    inlet_all_idx_idx = np.argsort(inlet_idx_mat)

    # don't try to understand this line (message from Jaap); KA: replaced empty "cell" contents with NaNs
    inlet_idx_mat[
        inlet_all_idx_idx[
            np.less_equal(
                np.diff(np.r_[inlet_all_idx[-1] - ny, inlet_all_idx]),
                1,
            )
        ]
    ] = np.nan
    inlet_idx_mat = inlet_idx_mat[~np.isnan(inlet_idx_mat)]
    # KA: here, inlet_idx is reduced to just the first index (still a list)
    inlet_idx = inlet_idx_mat.astype(int).tolist()

    return inlet_idx, inlet_idx_mat

def fluid_mechanics(
    inlet_idx,
    inlet_idx_mat,
    ny,
    dy,
    omega0,
    w,
    a0,
    man_n,
    d_b,
    marsh_cover,
    basin_width,
    min_inlet_separation=10000,
):
    r"""

    Parameters
    ----------
    inlet_idx: list of int
        Indices of inlet locations
    ny: int
        Number of alongshore cells

    Returns
    -------
    array of integers
        inlet_idx: indices of all inlets
    """

    inlet_asp = np.sqrt(0.005)  # aspect ratio inlet
    u_e = 1  # inlet equilibrium velocity [m/s]
    u_e_star = u_e / np.sqrt(g * a0)  # equilibrium inlet velocity (non-dimensional)

    # sort inlets (first index only) and find respective tidal prisms
    inlet_all_idx = np.sort(inlet_idx)
    inlet_all_idx_idx = np.argsort(inlet_idx)
    inlet_dist = np.diff(
        np.r_[
            inlet_all_idx[-1] - ny,
            inlet_all_idx,
            inlet_all_idx[0] + ny,
        ]
    )  # distance between inlets
    basin_length = np.minimum(
        min_inlet_separation,
        (dy * 0.5 * (inlet_dist[0:-1] + inlet_dist[1 : len(inlet_dist)])),
    )

    # see swart zimmerman
    ah_star = omega0 * w[inlet_idx] / np.sqrt(g * a0)
    c_d = g * man_n ** 2 / (d_b[inlet_idx] ** (1 / 3))
    gam = np.maximum(
        1e-3,
        inlet_asp
        * (
            (omega0 ** 2)
            * (1 - marsh_cover) ** 2
            * (basin_length[inlet_all_idx_idx] ** 2)
            * (basin_width[inlet_idx] ** 2)
            * a0
            / g
        )
        ** (1 / 4)
        / ((8 / 3 / np.pi) * c_d * w[inlet_idx]),
    )
    a_star_eq = a_star_eq_fun(ah_star, gam, u_e_star)
    u_eq = np.real(u(a_star_eq, gam, ah_star))
    ai_eq = (
        omega0
        * (1 - marsh_cover)
        * basin_length[inlet_all_idx_idx]
        * basin_width[inlet_idx]
        * np.sqrt(a0 / g)
    ) * a_star_eq  # KA: does it matter that this was last defined during the Tstorm year?

    # keep inlet open if velocity is at equilibrium (Escoffier); add
    # margin of 0.05 m/s for rounding errors etc
    inlet_close = np.logical_and(
        np.logical_or(np.less(u_eq, (u_e - 0.05)), np.isnan(u_eq)),
        np.greater(w[inlet_idx], 0),
    )

    # we don't have to think about this one every again!
    inlet_idx_mat[inlet_close] = np.nan  # KA: use inlet_idx_mat b/c float
    inlet_idx_close_mat = np.argwhere(np.isnan(inlet_idx_mat))  # KA: get index
    inlet_idx_mat = inlet_idx_mat[~np.isnan(inlet_idx_mat)]
    # KA: again here, inlet_idx is just the first index (still a list), and not sorted
    inlet_idx = inlet_idx_mat.astype(int).tolist()
    ai_eq[inlet_close] = np.nan
    ai_eq = ai_eq[~np.isnan(ai_eq)]

    wi_eq = np.sqrt(ai_eq) / inlet_asp  # calculate width and depths
    di_eq = ai_eq / wi_eq
    wi_cell = np.ceil(wi_eq / dy).astype(int)  # get cell widths per inlet
    return inlet_idx, inlet_idx_close_mat, wi_cell, di_eq, ai_eq, wi_eq

def inlet_morphodynamics(
    inlet_idx,
    new_inlet,
    time_index,
    wi_cell,
    ny,
    dy,
    x_b_fld_dt,
    w,
    Qs,
    h_b,
    di_eq,
    d_b,
    Qinlet,
    rho_w,
    ai_eq,
    wi_eq,
    wave_height,
    x_b,
    x_s,
    x_s_dt,
    w_b_crit,
    omega0,
    inlet_y,
    inlet_age,
    d_sf,
    u_e=1
    ):

    r"""

    Parameters
    ----------
    inlet_idx: list of int
        Indices of inlet locations
    new_inlet: int
        Index of the newest inlet
    ny: int
        Number of alongshore cells
    dy: float/int
        length of alongshore section
    wi_cell: int (list?)
        Width of cell per inlet
    x_b_fld_dt
    h_b: float?
        Barrier Height
    di_eq
    d_b
    Qinlet,
    rho_w: float
        density of sea water
    u_e: int/float
        inlet equilibrium velocity [m/s]
    ai_eq,
    wi_eq,
    wave_height,
    x_b,
    x_s,
    x_s_dt,
    w_b_crit:
        critical barrier width (for drowning?)
    omega0:
        tide frequency
    inlet_y,
    inlet_age,
    d_sf

    Returns
    -------
    array of integers
        inlet_idx: indices of all inlets
    """

    # KA: python object arrays to "mimic" Matlab cells for inlet tracking
    # in retrospect, probably didn't need objects. Empty list would have been fine.
    # LVB: got rid of object type specification
    inlet_nex = np.empty(np.size(inlet_idx))
    inlet_prv = np.empty(np.size(inlet_idx))

    # preallocate arrays for inlet migration and fractions based on I
    migr_up, delta, beta, beta_r, alpha, alpha_r, delta_r, Qs_in = [
        np.zeros(np.size(inlet_idx)).astype(float) for _ in range(8)
    ]

    # inlet morphodynamics per inlet (KA: again, j-1 here for python)
    for j in np.arange(1, np.size(inlet_idx) + 1):

        # breach sediment is added to the flood-tidal delta
        if (
                new_inlet.size > 0
                and inlet_idx[j - 1] == new_inlet
        ):  # KA: for python, need to check that the array isn't empty
            # KA: here, Jaap allows the indexing to wrap such that a new
            # inlet formed at the end of the model domain can deposit sediment
            # at the start of the model domain; does this wrapping for all
            # inlet dynamics (see other np.mods throughout code)
            new_inlet_idx = np.mod(
                new_inlet + np.r_[1: (wi_cell[j - 1] + 1)] - 1, ny
            )
            x_b_fld_dt[new_inlet_idx] = x_b_fld_dt[
                  new_inlet_idx
              ] + (
                      (h_b[new_inlet] + di_eq[j - 1]) * w[new_inlet]
              ) / (
                  d_b[new_inlet]
              )

            Qinlet[time_index - 1] = Qinlet[
             time_index - 1
             ] + (
                 (h_b[new_inlet] + d_b[new_inlet])
                 * w[new_inlet]
                 * wi_cell[j - 1]
                 * dy
             )

        # alongshore flux brought into inlet

        # Qs was calculated in the ast_model_on (line 781) but that was commented out in Katherine's BRIE
        Qs_in[j - 1] = Qs[inlet_idx[j - 1]]

        # find cells of inlet, updrift barrier, and downdrift barrier
        # KA: here, inlet_idx becomes a list of arrays, and again,
        # inlets wrap around the edges
        inlet_idx[j - 1] = np.mod(
            inlet_idx[j - 1] + np.r_[1: (wi_cell[j - 1] + 1)] - 1,
            ny,
        ).astype(int)
        inlet_nex[j - 1] = np.mod(inlet_idx[j - 1][-1] + 1, ny)
        inlet_prv[j - 1] = np.mod(inlet_idx[j - 1][0] - 1, ny)

        # find momentum balance of inlet to determine sediment
        # distribution fractions
        Mt = rho_w * u_e * u_e * ai_eq[j - 1]
        Mw = rho_w / 16 * g * wave_height ** 2 * wi_eq[j - 1]
        I = Mt / Mw * wi_eq[j - 1] / w[inlet_idx[j - 1][0]]
        h_b[inlet_idx[j - 1]] = 0

        # constrain to not widen
        Ab_prv = w[inlet_prv[j - 1]] * (
                h_b[inlet_idx[j - 1][0]] + di_eq[j - 1]
        )
        Ab_nex = w[inlet_nex[j - 1]] * (
                h_b[inlet_nex[j - 1]] + di_eq[j - 1]
        )

        # do fld delta eq volume
        Vfld = (
                (
                        x_b[inlet_idx[j - 1][0]]
                        - x_s[inlet_idx[j - 1][0]]
                        + w_b_crit
                )
                * wi_eq[j - 1]
                * d_b[inlet_idx[j - 1][0]]
        )
        Vfld_max = 1e4 * (u_e * ai_eq[j - 1] / 2 / omega0) ** 0.37

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

        delta[j - 1] = inlet_fraction(0, 1, 3, -3, I)
        beta[j - 1] = inlet_fraction(0, 1, 10, 3, I)
        beta_r[j - 1] = inlet_fraction(0, 0.9, 0.9, -3, I)

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
            inlet_prv[j - 1], inlet_idx[j - 1], inlet_nex[j - 1]
        ]

        x_b_fld_dt[temp_idx] = x_b_fld_dt[temp_idx] + fld_delta / (
                np.size(temp_idx) * dy
        ) / (h_b[temp_idx] + d_b[temp_idx])

        # migrate inlet indices (in m/dt)
        migr_up[j - 1] = Qs_in[j - 1] * (alpha_r[j - 1] + alpha[j - 1]) / Ab_prv
        #                migr_dw = (
        #                    Qs_in[j - 1]
        #                    * (alpha_r[j - 1] + beta_r[j - 1] + delta_r[j - 1])
        #                    / Ab_nex
        #                )

        # calculate where in the grid cell the inlet is, and add the
        # fractional migration to it
        inlet_y[inlet_idx[j - 1][0]] = (
                inlet_y[inlet_idx[j - 1][0]] + migr_up[j - 1] / dy
        )

        # how far are the inlets in their gridcell?
        # (or is inlet_y>1 or <0 and should the inlet hop one grid cell?)
        migr_int = np.floor(
            inlet_y[inlet_idx[j - 1][0]]
        )  # KA: unsure about these too
        migr_res = np.mod(inlet_y[inlet_idx[j - 1][0]], 1)

        # reset old grid cell
        inlet_y[inlet_idx[j - 1][0]] = 0

        # move inlet in gridcell
        inlet_idx[j - 1] = np.mod(
            inlet_idx[j - 1] + migr_int, ny
        ).astype(int)

        inlet_y[inlet_idx[j - 1][0]] = migr_res

        # how much q flood tidal delta in total
        Qinlet[time_index - 1] = (
                Qinlet[time_index - 1] + inlet_sink
        )  # m3 per time step

        # add inlet sink to shoreline change (updated May 27, 2020 so that shoreline change from inlet sink
        # now spread out along width of inlet +1 cell in both directions)
        # self._x_s_dt[inlet_nex[j - 1]] = (
        #         self._x_s_dt[inlet_nex[j - 1]]
        #         + inlet_sink / (self._h_b[inlet_nex[j - 1]] + self._d_sf) / self._dy
        # )
        x_s_dt[temp_idx] = (
                x_s_dt[temp_idx]
                + inlet_sink
                / (h_b[temp_idx] + d_sf)
                / len(temp_idx)
                / dy
        )

        # inlet age
        # fancy lightweight way to keep track of where inlets are in the model
        # KA: note that this differs from matlab version, here we do this all
        # in the for loop (but still [time step, inlet starting ID])
        inlet_age.append(  # KA: shouldn't this be time_index-1?
            [time_index, inlet_idx[j - 1][0].astype("int32")]
        )

    # reset arrays
    new_inlet = np.array([])
    return inlet_idx, migr_up, delta, beta, alpha, Qs_in

def inlet_statistics(
    time_index,
    dtsave,
    inlet_nr,
    inlet_idx,
    inlet_migr,
    migr_up,
    delta,
    inlet_delta,
    beta,
    inlet_beta,
    alpha,
    inlet_alpha,
    Qs_in,
    inlet_Qs_in,
    ai_eq,
    inlet_ai,
    dt):

    if (
            np.mod(time_index, dtsave) == 0
    ):  # KA: modified this from matlab version so that I can save every time step in python
        # skip first time step (initial condition)
        inlet_nr[
            np.fix(time_index / dtsave).astype(int) - 1
            ] = len(
            inlet_idx
        )  # number of inlets
        inlet_migr[
            np.fix(time_index / dtsave).astype(int) - 1
            ] = np.mean(migr_up / dt)

        if np.size(inlet_idx) != 0:
            inlet_Qs_in[
                np.fix(time_index / dtsave).astype(int) - 1
                ] = np.mean(Qs_in)
            inlet_alpha[
                np.fix(time_index / dtsave).astype(int) - 1
                ] = np.mean(alpha)
            inlet_beta[
                np.fix(time_index / dtsave).astype(int) - 1
                ] = np.mean(beta)
            inlet_delta[
                np.fix(time_index / dtsave).astype(int) - 1
                ] = np.mean(delta)
            inlet_ai[
                np.fix(time_index / dtsave).astype(int) - 1
                ] = np.mean(ai_eq)
    return inlet_nr, inlet_migr

class InletSpinner:
    """Transport sediment along a coast.

    Examples
    --------
    >>> from brie.lexi_inlet_spinner import InletSpinner
    >>> inlets = InletSpinner([0.0, 0.0, 1.0, 0.0, 0.0])
    >>> inlets.update()
    """

    K = calc_alongshore_transport_k()

    def __init__(
        self,
        shoreline_x,
        bay_shoreline_x,
        number_time_steps,
        save_spacing,
        basin_width=None,
        inlet_storm_frequency=10,
        barrier_height=2.0,
        alongshore_section_length=1.0,
        time_step=1.0,
        change_in_shoreline_x=0.0,
        wave_height=1.0,
        wave_period=10.0,
        wave_angle=0.0,
        wave_distribution=None,
        inlet_min_spacing=10000.0,
        tide_amplitude=0.5,
        tide_frequency=1.4e-4,
        lagoon_manning_n=0.05,
        back_barrier_marsh_fraction=0.5,
    ):
        """The InletSpinner module.

        Parameters
        ----------
        shoreline_x: float
            A shoreline position [m].
        alongshore_section_length: float, optional
            Length of each alongshore section [m].
        time_step: float, optional
            Time step of the numerical model [y].
        change_in_shoreline_x: float or array of float, optional
            Change in shoreline x position (accretion/erosion) [m].
        wave_height: float, optional
            Mean offshore significant wave height [m].
        wave_period: float, optional
            Mean wave period [s].
        wave_angle: float or array of float, optional
            Incoming wave angle relative to local shoreline normals [rad]. That is, a
            value of 0 means approaching waves are normal to the coast, negative
            values means waves approaching from the right, and positive from
            the left [deg]
        wave_distribution: a scipy distribution
        """

        self._shoreline_x = np.asarray(shoreline_x, dtype=float)
        self._bay_shoreline_x = np.asarray(bay_shoreline_x, dtype=float)
        self._h_b = barrier_height
        self._dy = alongshore_section_length
        self._dt = time_step
        self._dtsave = save_spacing  # LVB
        self._dx_dt = change_in_shoreline_x
        self._wave_height = wave_height
        self._wave_period = wave_period
        self._wave_angle = wave_angle
        self._x_b = bay_shoreline_x
        # self._h_b = barrier_height    #LVB: repeated
        self._nt = number_time_steps
        self._inlet_storm_frequency = inlet_storm_frequency
        self._create_inlet_now = False  # KA: added this boolean so we could be more flexible about when to add an inlet
        self._basin_width = basin_width

        # added from brie
        self._Jmin = inlet_min_spacing
        self._a0 = tide_amplitude
        self._omega0 = tide_frequency
        # self._inlet_asp = np.sqrt(0.005)  # aspect ratio inlet
        self._man_n = lagoon_manning_n
        # self._u_e = 1  # inlet equilibrium velocity [m/s]
        self._inlet_max = 100  # maximum number of inlets (mostly for debugging)
        self._marsh_cover = back_barrier_marsh_fraction

        # inlet dependednt variables added by LVB
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

        # other dependent variables
        self._ny = int(np.size(self._shoreline_x))
        # self._u_e_star = self._u_e / np.sqrt(
        #     self._g * self._a0
        # )  # equilibrium inlet velocity (non-dimensional)
        self._t = np.arange(
            self._dt, (self._dt * self._nt) + self._dt, self._dt
        )  # time array, KA: note, if we can eliminate this variable, that would be great (i.e., we wouldn't need nt)

        # initialize empty arrays
        self._inlet_idx = []
        # self._inlet_idx_mat = np.array([]).astype(
        #     float
        # )  # KA: we use this variable for NaN operations
        # self._inlet_idx_close_mat = np.array([])
        self._inlet_y = np.zeros(self._ny)
        self._barrier_volume = np.array([])
        self._q_s = np.empty_like(self._shoreline_x)
        self._x_b_fld_dt = np.zeros(self._ny)

        if wave_distribution is None:
            wave_distribution = scipy.stats.uniform(loc=-np.pi / 2.0, scale=np.pi)
        self._wave_distribution = wave_distribution

        # initial shoreline angles
        self._shoreline_angles = calc_shoreline_angles(
            self._shoreline_x, spacing=self._dy
        )

        self._time = 0.0

    def update(self):

        self._time += self._dt

        # if wave angle not given, pull from distribution
        if self._wave_angle is None:
            self._wave_angle = self._wave_distribution.rvs(size=1)

        # calculate shoreline angle for each time step
        calc_shoreline_angles(self._shoreline_x, self._dy, out=self._shoreline_angles)

        # sediment transport into inlets  KA: temporary placement until I figure out where this goes
        self._q_s[:] = (
            calc_inlet_alongshore_transport(
                self._wave_angle,
                shoreline_angle=self._shoreline_angles,
                wave_height=self._wave_height,
                wave_period=self._wave_period,
            )
            * self._dt
        )

        # -------------------------------

        self._x_b_fld_dt = np.zeros(self._ny)  # reset array of flood tidal deltas

        w = self._x_b - self._shoreline_x  # barrier width
        self._barrier_volume = (
            w * (self._h_b + 2) * np.sign(np.minimum(w, self._h_b))
        )  # barrier volume = barrier width times height + estimated inlet depth (KA: is inlet depth 2 m?)

        # where there is currently an inlet, set the barrier volume at that location to infinity
        if (
            np.size(self._inlet_idx) != 0
        ):  # KA: inlet_idx is a list here with arrays of different sizes (from previous time loop)
            self._barrier_volume[np.hstack(self._inlet_idx)] = np.inf
            self._inlet_idx.append(
                np.nonzero(self._barrier_volume < 0)[0]
            )  # add drowned cells to list of inlets

        # create a new inlet every # years, unless at max # inlets, or if boolean says to create inlet this year
        if (
            np.mod(self._t[self._time - 1], self._inlet_storm_frequency)
            < (self._dt / 2)
            and np.size(self._inlet_idx) < self._inlet_max
            or self._create_inlet_now  # NOTE TO ERIC: there is probably a more elegant way to do this
        ):
            self._inlet_idx, new_inlet = create_inlet(
                self._inlet_idx, self._Jmin, self._ny, self._dy, self._barrier_volume
            )

        if np.size(self._inlet_idx) != 0:
            self._inlet_idx, self._inlet_idx_mat = organize_inlet(
                self._inlet_idx, self._ny
            )  # get rid of duplicates and neighbours
            self._inlet_idx, self._inlet_idx_close_mat, wi_cell, di_eq, ai_eq, wi_eq = fluid_mechanics(
                self._inlet_idx, self._inlet_idx_mat, self._ny, self._dy, self._omega0, self._w, self._a0,
                self._man_n, self._d_b, self._marsh_cover, self._basin_width, min_inlet_separation=10000
            )  # do "fluid mechanics" of inlets
            # in paper they do sediment transport next, but I think it is okay to do it whenever
            self._inlet_idx, migr_up, delta, beta, alpha, Qs_in = inlet_morphodynamics(
                self._inlet_idx, new_inlet, self._time_index, wi_cell, self._ny, self._dy, self._x_b_fld_dt, w,
                self._q_s, self._h_b, di_eq, self._d_b, self._Qinlet, self._rho_w, ai_eq, wi_eq, self._wave_height,
                self._x_b, self._x_s, self._x_s_dt, self._w_b_crit, self._omega0, self._inlet_y, self._inlet_age,
                self._d_sf
            )  # inlet morphodynamics
            self._inlet_rn, self._inlet_migr = inlet_statistics(
                self._time, self._dtsave,self._inlet_nr, self._inlet_idx, self._inlet_migr, migr_up, delta,
                self._inlet_delta, beta, self._inlet_beta, alpha, self._inlet_alpha, Qs_in, self._inlet_Qs_in,
                ai_eq, self._inlet_ai, self._dt
            )  # inlet statistics


    @property
    def wave_angle(self):
        return self._wave_angle

    @property
    def q_s(self):
        return self._q_s

    @wave_angle.setter
    def wave_angle(self, new_val):
        if np.abs(new_val) <= np.deg2rad(90):
            raise ValueError("wave angle must be between -pi/2 and pi/2")
        self._wave_angle = new_val

    @property
    def shoreline_x(self):
        return self._shoreline_x

    @shoreline_x.setter
    def shoreline_x(self, new_val):
        self._shoreline_x = new_val

    @property
    def bay_shoreline_x(self):
        return self._bay_shoreline_x

    @bay_shoreline_x.setter
    def bay_shoreline_x(self, new_val):
        self._bay_shoreline_x = new_val

    @property
    def h_b(self):
        return self._h_b

    @h_b.setter
    def h_b(self, new_val):
        self._h_b = new_val

    @property
    def basin_width(self):
        return self._basin_width

    @basin_width.setter
    def basin_width(self, new_val):
        self._basin_width = new_val
