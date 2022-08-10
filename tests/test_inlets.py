import numpy as np
import pytest
import scipy.constants
g = scipy.constants.g
from numpy.testing import assert_array_almost_equal

from brie.lexi_inlet_spinner import (
    organize_inlet,
    fluid_mechanics,
    inlet_morphodynamics,
    inlet_statistics
)


def test_u(a_star=0, gam=0, ah_star=0, a0=0):
    result = np.sqrt(g * a0) * np.sqrt(
        gam
        / 2.0
        * np.sqrt(a_star)
        * (
            (-gam * np.sqrt(a_star) * ((a_star - ah_star) ** 2))
            + np.sqrt((gam ** 2) * a_star * ((a_star - ah_star) ** 4) + 4)
        )
    )
    assert result == 0


def test_organize_inlet():
    inlet_idx = [5, 5, 6, 100, 200]
    ny = 10
    inlet_array1, inlet_mat1 = organize_inlet(inlet_idx, ny)
    # code directly from master brie
    # get rid of duplicates and neighbours
    if np.size(inlet_idx) != 0:
        # KA: inlet_idx_mat is just inlet_idx concatenated into a single
        # array and made a float so we can use NaNs
        inlet_idx_mat = np.hstack(inlet_idx).astype(float)
        inlet_all_idx = np.sort(inlet_idx_mat)
        inlet_all_idx_idx = np.argsort(inlet_idx_mat)

        # don't try to understand this line (message from Jaap)
        # KA: replaced empty "cell" contents with NaNs
        inlet_idx_mat[
            inlet_all_idx_idx[
                np.less_equal(
                    np.diff(np.r_[inlet_all_idx[-1] - ny, inlet_all_idx]),
                    1,
                )
            ]
        ] = np.nan
        inlet_idx_mat = inlet_idx_mat[
            ~np.isnan(inlet_idx_mat)
        ]
        # KA: here, inlet_idx is reduced to just the first index (still a list)
        inlet_idx = inlet_idx_mat.astype(int).tolist()
    result = np.array_equal(inlet_array1, inlet_idx)
    assert result == True

# def test_fluids():
#     inlet_idx = [100, 150, 200]
#     ny = 10
#     inlet_idx, inlet_mat = organize_inlet(inlet_idx, ny)
#     dy = 1
#     tide_frequency = 1.4e-4
#     tide_amplitude = 0.5
#     man_n = 0.05
#     marsh_cover = 0.5
#     fluid_mechanics(inlet_idx, ny, dy, omega=tide_frequency, w, a0=tide_amplitude, man_n=man_n, d_b, marsh_cover, basin_width)