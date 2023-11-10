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

        traction = torch.tensor(traction_points.value)
        dot_product = torch.sum(traction * u_points, dim=1)

        left_lerp, right_lerp = 0.0, 0.0

        # if we are not on the left edge, add left lerp correction
        if traction_points.indices[0] % traction_points.width != 0:
            ul = u[traction_points.indices[0] - traction_points.stride]
            left_product = torch.sum(traction * ul)

            lt = traction_points.left_error / ds
            left_lerp = (
                traction_points.left_error
                * (left_product * lt + dot_product[0] * (2 - lt))
                / 2
            )

        # if we are not on the right edge, add right lerp correction
        if (traction_points.indices[-1] + 1) % traction_points.width != 0:
            ur = u[traction_points.indices[-1] + traction_points.stride]
            right_product = torch.sum(traction * ur)

            rt = traction_points.right_error / ds
            right_lerp = (
                traction_points.right_error
                * (right_product * rt + dot_product[-1] * (2 - rt))
                / 2
            )

        return left_lerp + right_lerp + torch.trapezoid(dot_product, dx=ds)

    return external_energy
