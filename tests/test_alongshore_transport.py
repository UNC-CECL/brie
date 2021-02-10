from functools import partial

import numpy as np
import pytest
import scipy.stats
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix

from brie.alongshore_transporter import (
    _build_matrix,
    _build_tridiagonal_matrix,
    calc_alongshore_transport_k,
    calc_coast_diffusivity,
    calc_coast_qs,
    calc_shoreline_angles,
    calc_inlet_alongshore_transport
)


def old_calc_shoreline_angles(x, spacing=1.0):
    x = np.asarray(x)
    theta = 180 * (np.arctan2((x[np.r_[1 : len(x), 0]] - x), spacing)) / np.pi
    return np.deg2rad(theta)


def old_calc_coast_qs(wave_angle, wave_height=1.0, wave_period=10.0):
    return (
        wave_height ** 2.4
        * (wave_period ** 0.2)
        * 3600
        * 365
        * 24
        * calc_alongshore_transport_k()
        * (np.cos(wave_angle) ** 1.2)
        * np.sin(wave_angle)
    )  # [m3/yr]


def old_calc_inlet_alongshore_transport(
    wave_angle, shoreline_angle=0.0, wave_height=1.0, wave_period=10.0
):
    dt = 1.0
    wave_climl = 180
    angle_array = np.deg2rad(np.linspace(-90.0, 90.0, wave_climl + 1))

    coast_qs = old_calc_coast_qs(
        angle_array, wave_height=wave_height, wave_period=wave_period
    )
    theta = np.rad2deg(shoreline_angle)
    wave_ang = 90 - np.rad2deg(wave_angle)

    q_s = (
        dt
        * coast_qs[
            np.minimum(
                wave_climl + 1,
                np.maximum(
                    1,
                    np.round(
                        # wave_climl
                        # - wave_ang
                        # - (wave_climl / 180 * theta)
                        # wave_climl + theta - wave_ang
                        wave_climl
                        - theta
                        - wave_ang
                        + 1
                    ),
                ),
            ).astype(int)
            - 1
        ]
    ).astype(float)

    return q_s


def old_calc_coast_diffusivity(
    wave_pdf, shoreline_angles, wave_height=1.0, wave_period=10.0, h_b_crit=2.0
):
    wave_climl = 180
    # angle_array, step = np.linspace(-90.0, 90.0, wave_climl + 1, retstep=True)
    angle_array, step = np.linspace(-89.5, 89.5, 180, retstep=True)
    angle_array = np.deg2rad(angle_array)

    d_sf = 8.9 * wave_height

    diff = (
            -(
                    calc_alongshore_transport_k()
                    / (h_b_crit + d_sf)
                    * wave_height ** 2.4
                    * wave_period ** 0.2
            )
            * 365
            * 24
            * 3600
            * (np.cos(angle_array) ** 0.2)
            * (1.2 * np.sin(angle_array) ** 2 - np.cos(angle_array) ** 2)
    )

    # KA NOTE: the "same" method differs in Matlab and Numpy; here we pad and slice out the "same" equivalent
    # conv = np.convolve(waves.pdf(np.deg2rad(angle_array)) * np.deg2rad(step), diff, mode="full")
    conv = np.convolve(wave_pdf(angle_array) * np.deg2rad(step), diff, mode="full")
    npad = len(diff) - 1
    first = npad - npad // 2
    coast_diff_phi0_theta = conv[first: first + len(angle_array)]

    theta = np.rad2deg(shoreline_angles)

    coast_diff = coast_diff_phi0_theta[
        np.maximum(
            0,
            np.minimum(wave_climl + 1, np.round(90 - theta).astype(int)),
        )
    ]  # is this the relative wave angle? note that this indexing doesn't work with bounds other than [-90,90]
    # so this coast_diff differs from the result of our interpolation in AlongshoreTransporter. It should work once we
    # implement the Ashton distribution

    return coast_diff, coast_diff_phi0_theta

def old_build_matrix(x_s, wave_distribution, dy=1.0, wave_height=1.0, wave_period=10.0, dt=1.0, x_s_dt=0):

    wave_climl = 180
    ny = len(x_s)
    h_b_crit = 2.0

    di = (
        np.r_[
            ny,
            np.arange(2, ny + 1),
            np.arange(1, ny + 1),
            np.arange(1, ny),
            1,
        ]
        - 1
    )
    dj = (
        np.r_[
            1,
            np.arange(1, ny),
            np.arange(1, ny + 1),
            np.arange(2, ny + 1),
            ny,
        ]
        - 1
    )

    theta = np.rad2deg(old_calc_shoreline_angles(x_s, spacing=dy))

    # angle_array, step = np.linspace(-90.0, 90.0, wave_climl + 1, retstep=True)
    # angle_array = np.deg2rad(angle_array)
    angle_array, step = np.linspace(-89.5, 89.5, 180, retstep=True)
    angle_array = np.deg2rad(angle_array)

    d_sf = 8.9 * wave_height

    diff=(
            -(
                    calc_alongshore_transport_k()
                    / (h_b_crit + d_sf)
                    * wave_height ** 2.4
                    * wave_period ** 0.2
            )
            * 365
            * 24
            * 3600
            * (np.cos(angle_array) ** 0.2)
            * (1.2 * np.sin(angle_array) ** 2 - np.cos(angle_array) ** 2)
    )

    # KA NOTE: the "same" method differs in Matlab and Numpy; here we pad and slice out the "same" equivalent
    conv = np.convolve(wave_distribution.pdf(angle_array) * np.deg2rad(step), diff, mode="full")
    npad = len(diff) - 1
    first = npad - npad // 2
    coast_diff = conv[first: first + len(angle_array)]

    r_ipl = np.maximum(
        0,
        (
            coast_diff[
                np.maximum(
                    1,
                    np.minimum(wave_climl + 1, np.round(90 - theta).astype(int)),
                )
            ]
            * dt
            / 2
            / dy ** 2
        ),
    )

    dv = np.r_[-r_ipl[-1], -r_ipl[1:], 1 + 2 * r_ipl, -r_ipl[0:-1], -r_ipl[0]]
    A = csr_matrix((dv, (di, dj)))

    RHS = (
        x_s
        + r_ipl
        * (
                x_s[np.r_[1:ny, 0]]
                - 2 * x_s
                + x_s[np.r_[ny - 1, 0 : ny - 1]]
        )
        + x_s_dt  # I think this was dropped just for testing
    )

    return A, RHS, r_ipl


def old_build_tridiag(data):
    ny = len(data)

    di = (
        np.r_[
            ny,
            np.arange(2, ny + 1),
            np.arange(1, ny + 1),
            np.arange(1, ny),
            1,
        ]
        - 1
    )
    dj = (
        np.r_[
            1,
            np.arange(1, ny),
            np.arange(1, ny + 1),
            np.arange(2, ny + 1),
            ny,
        ]
        - 1
    )

    # dv = np.r_[-r_ipl[-1], -r_ipl[1:], 1 + 2 * r_ipl, -r_ipl[0:-1], -r_ipl[0]]

    dv = np.r_[data[-1], data[1:], data, data[0:-1], data[0]]
    A = csr_matrix((dv, (di, dj)))

    return A


@pytest.mark.parametrize("angle", (60, 45, 30, 0, -30, -45, -60.0))
@pytest.mark.parametrize("shoreline_angle", (-15, 0, 15))
def test_inlet_alongshore_transport_old_to_new(angle, shoreline_angle):
    angle = np.deg2rad(angle)
    shoreline_angle = np.deg2rad(shoreline_angle)
    assert_array_almost_equal(
        old_calc_inlet_alongshore_transport(
            angle, shoreline_angle=np.full(5, shoreline_angle)
        ),
        calc_inlet_alongshore_transport(angle, shoreline_angle=np.full(5, shoreline_angle)),
    )


def test_shoreline_old_to_new():
    spacing = 100.0
    x = np.random.uniform(low=-10.0, high=10.0, size=1000)
    assert_array_almost_equal(
        calc_shoreline_angles(x, spacing=spacing),
        old_calc_shoreline_angles(x, spacing=spacing),
    )

    x = np.arange(12.0)
    assert_array_almost_equal(
        calc_shoreline_angles(x, spacing=1.0), old_calc_shoreline_angles(x, spacing=1.0)
    )


@pytest.mark.parametrize("func", (calc_shoreline_angles, old_calc_shoreline_angles))
def test_default_spacing(func):
    x = np.arange(5)
    assert np.all(func(x, spacing=1.0) == func(x))


@pytest.mark.parametrize("func", (calc_shoreline_angles, old_calc_shoreline_angles))
@pytest.mark.parametrize("spacing", [1.0, 0.5, -0.5])
def test_spacing(func, spacing):
    x = np.arange(5)
    angle = np.arctan2(1.0, spacing)
    assert func(x, spacing=spacing)[:-1] == pytest.approx(angle)
    assert func(-x, spacing=spacing)[:-1] == pytest.approx(-angle)


@pytest.mark.parametrize("func", (calc_shoreline_angles, old_calc_shoreline_angles))
def test_flat_shoreline(func):
    assert func(np.full(5, 0.0)) == pytest.approx(0.0)
    assert func(np.full(5, 1.0)) == pytest.approx(0.0)
    assert func(np.full(5, -1.0)) == pytest.approx(0.0)


@pytest.mark.parametrize("func", (calc_shoreline_angles, old_calc_shoreline_angles))
def test_wraparound(func):
    assert np.all(func([1.0, 0.0, 0.0, 0.0]) == [-np.pi / 4.0, 0.0, 0.0, np.pi / 4.0])


def test_calc_coast_qs_old_to_new():
    angles = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=1000)
    assert_array_almost_equal(
        calc_coast_qs(angles, wave_height=2.0, wave_period=5.0),
        old_calc_coast_qs(angles, wave_height=2.0, wave_period=5.0),
    )


@pytest.mark.parametrize("func", (calc_coast_qs, old_calc_coast_qs))
def test_calc_coast_qs_normal_waves(func):
    assert func(0.0) == pytest.approx(0.0)
    assert np.all(func([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0])


@pytest.mark.parametrize("func", (calc_coast_qs, old_calc_coast_qs))
def test_calc_coast_qs_left_transport(func):
    angles = np.random.uniform(low=-np.pi / 2.0, high=0.0, size=50)
    assert func(-np.pi / 4.0) < 0.0


@pytest.mark.parametrize("func", (calc_coast_qs, old_calc_coast_qs))
def test_calc_coast_qs_right_transport(func):
    angles = np.random.uniform(low=0.0, high=np.pi / 2.0, size=50)
    assert func(np.pi / 4.0) > 0.0


@pytest.mark.parametrize("func", (calc_coast_qs, old_calc_coast_qs))
def test_calc_coast_qs_transport_is_symmetrical(func):
    angles = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=50)
    assert np.all(func(angles) == pytest.approx(-func(-angles)))


@pytest.mark.parametrize("func", (calc_coast_qs, old_calc_coast_qs))
def test_calc_coast_qs_wave_height(func):
    angles = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=50)
    assert np.all(func(angles) == func(angles, wave_height=1.0))

    assert np.all(np.abs(func(angles, wave_height=2.0)) > np.abs(func(angles)))
    assert np.all(np.abs(func(angles, wave_height=0.5)) < np.abs(func(angles)))


@pytest.mark.parametrize("func", (calc_coast_qs, old_calc_coast_qs))
def test_calc_coast_qs_wave_period(func):
    angles = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=50)
    assert np.all(func(angles) == calc_coast_qs(angles, wave_period=10.0))

    assert np.all(np.abs(func(angles, wave_period=20.0)) > np.abs(func(angles)))
    assert np.all(np.abs(func(angles, wave_period=5.0)) < np.abs(func(angles)))


@pytest.mark.parametrize(
    "func", (calc_inlet_alongshore_transport, old_calc_inlet_alongshore_transport)
)
def test_alongshore_transport(func):
    # angles = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=50)
    # angles = np.random.uniform(low=0.0, high=np.pi, size=50)

    angles = np.deg2rad(np.linspace(-90, 90, 181))

    # assert_array_almost_equal(func(angles), calc_coast_qs(np.pi / 2.0 - angles))
    assert_array_almost_equal(func(angles), calc_coast_qs(angles))
    assert_array_almost_equal(func(angles), func(angles, shoreline_angle=0.0))


@pytest.mark.parametrize("shoreline_angle", (-15, 0, 15))
@pytest.mark.parametrize(
    "func", (calc_inlet_alongshore_transport, old_calc_inlet_alongshore_transport)
)
def test_alongshore_transport_shoreline_angle(func, shoreline_angle):
    shoreline_angle = np.deg2rad(shoreline_angle)
    assert_array_almost_equal(
        func(0.0, shoreline_angle=shoreline_angle),
        func(-shoreline_angle, shoreline_angle=0.0),
    )


@pytest.mark.parametrize(
    "func", (calc_inlet_alongshore_transport, old_calc_inlet_alongshore_transport)
)
def test_alongshore_transport_normal_waves(func):
    assert func(np.pi / 4.0, shoreline_angle=np.pi / 4.0) == pytest.approx(0.0)
    assert func(-np.pi / 4.0, shoreline_angle=-np.pi / 4.0) == pytest.approx(0.0)

    angles = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=50)
    assert func(angles, shoreline_angle=angles) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "func", (calc_inlet_alongshore_transport, old_calc_inlet_alongshore_transport)
)
def test_alongshore_transport_symmetrical(func):
    angles = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=50)
    assert_array_almost_equal(
        func(0.0, shoreline_angle=angles), -func(0.0, shoreline_angle=-angles)
    )


@pytest.mark.parametrize(
    "func", (calc_inlet_alongshore_transport, old_calc_inlet_alongshore_transport)
)
def test_alongshore_transport_to_the_left(func):
    angles = np.random.uniform(low=0.0, high=np.pi / 2.0, size=50)
    assert np.all(
        func(0.0, shoreline_angle=angles)
        <= 0.0
        # calc_alongshore_transport(np.pi / 2.0, shoreline_angle=angles) < 0.0
    )


@pytest.mark.parametrize(
    "func", (calc_inlet_alongshore_transport, old_calc_inlet_alongshore_transport)
)
def test_alongshore_transport_to_the_right(func):
    angles = np.random.uniform(low=-np.pi / 2.0, high=0.0, size=50)
    assert np.all(func(0.0, shoreline_angle=angles) >= 0.0)
    # assert np.all(calc_alongshore_transport(np.pi / 2.0, shoreline_angle=angles) > 0.0)


@pytest.mark.parametrize(
    "func", (calc_inlet_alongshore_transport, old_calc_inlet_alongshore_transport)
)
def test_alongshore_transport_parallel(func):
    angles = np.random.uniform(low=-np.pi / 2.0, high=-np.pi / 4.0, size=50)
    assert_array_almost_equal(
        func(angles, shoreline_angle=np.pi / 4.0), func(0.0, shoreline_angle=0.0)
    )

    angles = np.random.uniform(low=np.pi / 4.0, high=np.pi / 2.0, size=50)
    assert_array_almost_equal(
        func(angles, shoreline_angle=-np.pi / 4.0), func(np.pi, shoreline_angle=0.0)
    )


@pytest.mark.parametrize("angle", (-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75))
def test_coast_diff_old_to_new(angle):
    dist = scipy.stats.uniform(loc=-np.pi / 2.0, scale=np.pi)
    coast_diff, dummy1 = calc_coast_diffusivity(dist.pdf, np.deg2rad(angle))
    old_coast_diff, dummy2 = old_calc_coast_diffusivity(dist.pdf, np.deg2rad(angle))
    assert coast_diff == pytest.approx(old_coast_diff)


@pytest.mark.parametrize("func", (calc_coast_diffusivity, old_calc_coast_diffusivity))
@pytest.mark.parametrize("angle", (-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75))
def test_coast_diff_uniform_always_positive(func, angle):
    dist = scipy.stats.uniform(loc=-np.pi / 2.0, scale=np.pi)
    qs, dummy = func(dist.pdf, np.deg2rad(angle))
    assert qs > 0.0


@pytest.mark.parametrize("func", (calc_coast_diffusivity, old_calc_coast_diffusivity))
def test_coast_diff_symmetrical(func):
    dist = scipy.stats.uniform(loc=-np.pi / 2.0, scale=np.pi)
    #angles = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=500)
    angles = np.linspace(-89.5, 89.5, 180)
    angles = np.deg2rad(angles)
    diff_pos, dummy = func(dist.pdf, angles)
    diff_neg, dummy = func(dist.pdf, -angles)
    assert_array_almost_equal(diff_pos, diff_neg)


def test_build_matrix():
    mat = _build_tridiagonal_matrix([1, 2, 3, 4, 5])
    assert np.all(
        mat.toarray()
        == [
            [1, 1, 0, 0, 1],
            [2, 2, 2, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 4, 4, 4],
            [5, 0, 0, 5, 5],
        ]
    )


def test_build_matrix_with_lower():
    mat = _build_tridiagonal_matrix([1, 2, 3, 4, 5], lower=[11, 12, 13, 14, 15])
    assert np.all(
        mat.toarray()
        == [
            [1, 1, 0, 0, 11],
            [12, 2, 2, 0, 0],
            [0, 13, 3, 3, 0],
            [0, 0, 14, 4, 4],
            [5, 0, 0, 15, 5],
        ]
    )


def test_build_matrix_with_upper():
    mat = _build_tridiagonal_matrix([1, 2, 3, 4, 5], upper=[11, 12, 13, 14, 15])
    assert np.all(
        mat.toarray()
        == [
            [1, 11, 0, 0, 1],
            [2, 2, 12, 0, 0],
            [0, 3, 3, 13, 0],
            [0, 0, 4, 4, 14],
            [15, 0, 0, 5, 5],
        ]
    )


def test_build_matrix_with_upper_and_lower():
    mat = _build_tridiagonal_matrix([1, 2, 3, 4, 5], lower=[0] * 5, upper=[0] * 5)
    assert np.all(
        mat.toarray()
        == [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )


def test_tridiag_old_and_new():
    data = np.arange(1, 6)
    assert_array_almost_equal(
        _build_tridiagonal_matrix(data).toarray(),
        old_build_tridiag(data).toarray(),
    )


def test_build_matrices_old_and_new():
    data = np.arange(1, 6)
    x_s = np.zeros(13)
    x_s[-1] = 1.0
    wave_distribution = scipy.stats.uniform(loc=-np.pi / 2.0, scale=np.pi)
    expected_mat, expected_rhs, expected_v = old_build_matrix(x_s, wave_distribution)
    actual_mat, actual_rhs, actual_v = _build_matrix(x_s, wave_distribution)

    assert_array_almost_equal(expected_v, actual_v)
    assert_array_almost_equal(expected_rhs, actual_rhs)
    assert_array_almost_equal(expected_mat.toarray(), actual_mat.toarray())
