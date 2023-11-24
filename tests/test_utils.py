import numpy as np

from src.utils import constrain


def test_constrain():
    assert constrain(np.pi, 1) == "3"
    assert constrain(np.pi, 2) == "3 "
    assert constrain(np.pi, 3) == "3.1"
    assert constrain(-1 / 173, 6) == "-6e-03"
    assert constrain(-1 / 173, 5) == "-0.00"
    assert constrain(5.0, 4) == "5.00"
    assert constrain(5, 4) == " 5  "
    assert constrain(5555555555, 6) == " 6e+09"
    assert constrain(5555555555, 4) == "Bigg"
    assert constrain("hi", 6) == "  hi  "
    assert constrain("funny", 2) == "fu"
