import torch


def calculate_external_energy(
    u: torch.Tensor,
    dxdy: tuple[float, float],
    neumanBC_idx: list[torch.Tensor],
    neumanBC_values: list[torch.Tensor],
):
    dx = dxdy[0]

    external_energy = torch.tensor(0.0)
    for idxs, values in zip(neumanBC_idx, neumanBC_values):
        neumannPt_u = u[idxs.cpu().numpy()]
        W_ext = torch.einsum("ij,ij->i", neumannPt_u, values) * dx

        W_ext[-1] = W_ext[-1] / 2
        W_ext[0] = W_ext[0] / 2

        external_energy += torch.sum(W_ext)

    return external_energy
