import numpy as np
import pytest
import scipy.constants
g = scipy.constants.g
from numpy.testing import assert_array_almost_equal

from brie.lexi_inlet_spinner import (
    create_inlet,
    organize_inlet,
    fluid_mechanics,
    inlet_morphodynamics,
    inlet_statistics
)

from brie.alongshore_transporter import (
    calc_alongshore_transport_k,
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


# def test_calc_coast(angles=np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=1000), wave_height=1.0, wave_period=10.0):
#     qs_array = wave_height ** 2.4 * (wave_period ** 0.2) * 3600 * 365 * 24 * calc_alongshore_transport_k() * \
#                (np.cos(angles) ** 1.2) * np.sin(angles)
#     assert_array_almost_equal(
#         calc_coast_qs(angles, wave_height, wave_period),
#         qs_array)

# def test_create_inlet():
#
