import numpy as np

from src.utils import constrain, prettify_seconds


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


def test_prettify_seconds():
    assert prettify_seconds(0.865255979) == "865ms"
    assert prettify_seconds(8.539734222) == "8s 540ms"
    assert prettify_seconds(85.01969522) == "1m 25s 20ms"
    assert prettify_seconds(20631.78786) == "5h 43m 51s 788ms"
    assert prettify_seconds(885479.7776) == "245h 57m 59s 778ms"
