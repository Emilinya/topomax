import numpy as np

from src.utils import constrain, Timer


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
    assert Timer.prettify_seconds(0.865255979432265) == "865ms"
    assert Timer.prettify_seconds(8.539734222673566) == "8s 540ms"
    assert Timer.prettify_seconds(85.01969522320721) == "1m 25s 20ms"
    assert Timer.prettify_seconds(20631.78786771343) == "5h 43m 51s 788ms"
    assert Timer.prettify_seconds(885479.7776801552) == "245h 57m 59s 778ms"
