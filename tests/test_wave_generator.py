import numpy as np
import pytest
from scipy.integrate import quad

from brie.waves import WaveAngleGenerator, ashton


VALID_ASYMMETRY = (0.0, 0.25, 0.5, 0.75, 1.0)
VALID_HIGH_FRACTION = (0.0, 0.25, 0.5, 0.75, 1.0)


@pytest.mark.parametrize("asymmetry", VALID_ASYMMETRY)
@pytest.mark.parametrize("high_fraction", VALID_HIGH_FRACTION)
@pytest.mark.parametrize("loc", (0, -1, 1))
@pytest.mark.parametrize("scale", (1, 2))
def test_ashton(loc, scale, asymmetry, high_fraction):
    numbers = ashton.rvs(size=10000, loc=loc, scale=scale, a=asymmetry, h=high_fraction)
    assert np.all(numbers <= loc + scale)
    assert np.all(numbers >= loc)


@pytest.mark.parametrize("asymmetry", VALID_ASYMMETRY)
@pytest.mark.parametrize("high_fraction", VALID_HIGH_FRACTION)
@pytest.mark.parametrize("loc", (0, -1, 1))
@pytest.mark.parametrize("scale", (1, 2))
def test_ashton_frozen(loc, scale, asymmetry, high_fraction):
    dist = ashton(loc=loc, scale=scale, a=asymmetry, h=high_fraction)
    assert np.all(dist.rvs(size=10000) <= loc + scale)
    assert np.all(dist.rvs(size=10000) >= loc)


@pytest.mark.parametrize(
    "asymmetry,high_fraction", ((-0.5, 0.5), (1.5, 0.5), (0.5, -0.5), (0.5, 1.5))
)
def test_ashton_invalid_shapes(asymmetry, high_fraction):
    with pytest.raises(ValueError):
        dist = ashton(a=asymmetry, h=high_fraction).rvs(size=1)

    with pytest.raises(ValueError):
        ashton.rvs(size=1, a=asymmetry, h=high_fraction)


@pytest.mark.parametrize("asymmetry", VALID_ASYMMETRY)
@pytest.mark.parametrize("high_fraction", VALID_HIGH_FRACTION)
@pytest.mark.parametrize("loc", (0, -1, 1))
@pytest.mark.parametrize("scale", (1, 2))
def test_ashton_pdf(loc, scale, asymmetry, high_fraction):
    dist = ashton(
        a=asymmetry, h=high_fraction, loc=loc, scale=scale
    )

    assert dist.pdf(-91) == pytest.approx(0.0)
    assert dist.pdf(91) == pytest.approx(0.0)
    
    y, abserr = quad(dist.pdf, loc, loc + scale)
    assert y == pytest.approx(1.0)


def test_ashton_cdf():
    dist = ashton(a=0.5, h=0.5, loc=-90, scale=180)
    assert dist.cdf(-90.0) == pytest.approx(0.0)
    assert dist.cdf(90.0) == pytest.approx(1.0)

    assert dist.cdf(-91.0) == pytest.approx(0.0)
    assert dist.cdf(91.0) == pytest.approx(1.0)

    assert dist.cdf(-45.0) == pytest.approx(0.25)
    assert dist.cdf(0.0) == pytest.approx(0.5)
    assert dist.cdf(45.0) == pytest.approx(0.75)

    dist = ashton(
        a=np.random.random(), h=np.random.random(), loc=-90, scale=180
    )
    assert dist.cdf(-90.0) == pytest.approx(0.0)
    assert dist.cdf(90.0) == pytest.approx(1.0)


def test_waves_min_and_max():

    waves = WaveAngleGenerator()
    angles = waves.next(10000)
    assert np.all(angles >= -np.pi / 2.0)
    assert np.all(angles <= np.pi / 2.0)


@pytest.mark.parametrize(
    "asymmetry,highness,in_bounds",
    [
        (1.0, 1.0, lambda x: (-90 <= x) & (x <= -45.0)),
        (1.0, 0.0, lambda x: (-45 <= x) & (x <= 0.0)),
        (0.0, 0.0, lambda x: (0 <= x) & (x <= 45.0)),
        (0.0, 1.0, lambda x: (45 <= x) & (x <= 90.0)),
        (0.5, 0.0, lambda x: (-45 <= x) & (x <= 45.0)),
        (0.5, 1.0, lambda x: ((-90 <= x) & (x <= -45.0)) | ((45 <= x) & (x <= 90))),
    ],
)
def test_angle_bounds(asymmetry, highness, in_bounds):
    waves = WaveAngleGenerator(asymmetry=asymmetry, high_fraction=highness)
    angles = waves.next(10000)
    assert np.all(in_bounds(np.rad2deg(angles)))


def test_bad_values():
    with pytest.raises(ValueError):
        WaveAngleGenerator(asymmetry=-0.1)
    with pytest.raises(ValueError):
        WaveAngleGenerator(asymmetry=1.1)
    with pytest.raises(ValueError):
        WaveAngleGenerator(high_fraction=-0.1)
    with pytest.raises(ValueError):
        WaveAngleGenerator(high_fraction=1.1)


def test_pdf():
    waves = WaveAngleGenerator(
        asymmetry=np.random.random(), high_fraction=np.random.random()
    )
    x, step = np.linspace(-np.pi / 2.0, np.pi / 2.0, 10000, retstep=True)
    # y = waves.pdf(x)

    # area_under_curve = np.trapz(y, x)
    # assert area_under_curve == pytest.approx(1.0, abs=step ** 2)
    
    area_under_curve, abserr = quad(waves.pdf, -np.pi / 2.0, np.pi / 2.0)
    assert area_under_curve == pytest.approx(1.0)


def test_cdf():
    waves = WaveAngleGenerator(asymmetry=0.5, high_fraction=0.5)
    assert waves.cdf(-np.pi / 2.0) == pytest.approx(0.0)
    assert waves.cdf(np.pi / 2.0) == pytest.approx(1.0)

    assert waves.cdf(-np.pi / 4.0) == pytest.approx(0.25)
    assert waves.cdf(0.0) == pytest.approx(0.5)
    assert waves.cdf(np.pi / 4.0) == pytest.approx(0.75)

    waves = WaveAngleGenerator(
        asymmetry=np.random.random(), high_fraction=np.random.random()
    )
    assert waves.cdf(-np.pi / 2.0) == pytest.approx(0.0)
    assert waves.cdf(np.pi / 2.0) == pytest.approx(1.0)
