import numpy as np
import pytest

from brie.brie import WaveAngleGenerator


def test_waves_min_and_max():

    waves = WaveAngleGenerator()
    angles = waves.next(10000)
    assert np.all(angles >= -90.0)
    assert np.all(angles <= 90.0)


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
    assert np.all(in_bounds(angles))


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
    x, step = np.linspace(-90, 90, 10000, retstep=True)
    y = waves.pdf(x)

    area_under_curve = np.trapz(y, x)
    assert area_under_curve == pytest.approx(1.0, abs=step ** 2)


def test_cdf():
    waves = WaveAngleGenerator(asymmetry=0.5, high_fraction=0.5)
    assert waves.cdf(-90.0) == pytest.approx(0.0)
    assert waves.cdf(90.0) == pytest.approx(1.0)

    assert waves.cdf(-45.0) == pytest.approx(0.25)
    assert waves.cdf(0.0) == pytest.approx(0.5)
    assert waves.cdf(45.0) == pytest.approx(0.75)

    waves = WaveAngleGenerator(
        asymmetry=np.random.random(), high_fraction=np.random.random()
    )
    assert waves.cdf(-90.0) == pytest.approx(0.0)
    assert waves.cdf(90.0) == pytest.approx(1.0)
