import numpy as np

from DEM_src.utils import flatten, unflatten


def test_flatten_and_unflatten():
    np.random.seed(2023)

    shape = (3, 4)
    for n in [2, 4]:
        arrs = [np.random.rand(*shape) for _ in range(n)]

        flat = flatten(arrs)
        assert flat.shape == (np.prod(shape), n)

        unflat = unflatten(flat, shape)
        assert unflat.shape == (n, *shape)

        for i, arr in enumerate(arrs):
            assert np.all(unflat[i, :, :] == arr)
