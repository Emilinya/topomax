import torch
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

        # unflatten can also operate on tensors!
        torch_unflat = unflatten(torch.from_numpy(flat), shape)
        assert torch_unflat.shape == (n, *shape)

        for i, arr in enumerate(arrs):
            assert torch.all(torch_unflat[i, :, :] == torch.from_numpy(arr))
