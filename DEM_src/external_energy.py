import torch

from DEM_src.bc_helpers import TractionPoints


def calculate_external_energy(
    u: torch.Tensor,
    dxdy: tuple[float, float],
    device: torch.device,
    traction_points_list: list[TractionPoints],
):
    external_energy = torch.tensor(0.0)
    for traction_points in traction_points_list:
        torch_traction = torch.from_numpy(traction_points.values).float().to(device)
        u_points = u[traction_points.indices]
        ds = dxdy[traction_points.side_index]

        external_energy += torch.sum(u_points * torch_traction) * ds

    return external_energy
