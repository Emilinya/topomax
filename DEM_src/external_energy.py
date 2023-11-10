import torch

from DEM_src.bc_helpers import TractionPoints


def calculate_external_energy(
    u: torch.Tensor,
    dxdy: tuple[float, float],
    traction_points_list: list[TractionPoints],
):
    external_energy = torch.tensor(0.0)
    for traction_points in traction_points_list:
        u_points = u[traction_points.indices]
        ds = dxdy[traction_points.side_index]

        dot_product = torch.sum(torch.tensor(traction_points.value) * u_points, dim=1)

        # the bug sure looks silly written like ths
        if len(dot_product) == 1:
            external_energy += dot_product[0] * ds / 4
        else:
            external_energy += torch.trapezoid(dot_product, dx=ds)

    return external_energy
