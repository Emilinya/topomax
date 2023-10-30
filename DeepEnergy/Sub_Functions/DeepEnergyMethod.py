import time

import torch
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from torch.autograd import grad

from Sub_Functions import utils as util
from Sub_Functions.StrainEnergy import StrainEnergy
from Sub_Functions.MultiLayerNet import MultiLayerNet
from Sub_Functions.external_energy import calculate_external_energy
from Sub_Functions.data_structs import Domain, TopOptParameters, NNParameters


class DeepEnergyMethod:
    # Instance attributes
    def __init__(
        self,
        device: torch.device,
        example: int,
        to_parameters: TopOptParameters,
        nn_parameters: NNParameters,
    ):
        # self.data = data
        self.model = MultiLayerNet(nn_parameters)
        self.model = self.model.to(device)

        self.loss_array = []
        self.device = device
        self.to_parameters = to_parameters
        self.nn_parameters = nn_parameters
        self.example = example

    def train_model(
        self,
        rho: npt.NDArray[np.float64],
        domain: Domain,
        iteration: int,
        neumannBC: dict[str, dict[str, npt.NDArray[np.float64] | float]],
        dirichletBC: dict[str, dict[str, npt.NDArray[np.float64] | float]],
        training_times: list[float],
        objective_values: list[float],
    ):
        x = torch.from_numpy(domain.coordinates).float()
        x = x.to(self.device)
        x.requires_grad_(True)

        density = torch.from_numpy(rho).float()
        density = torch.reshape(density, [domain.Ny - 1, domain.Nx - 1]).to(self.device)

        neumannBC_coordinates: list[torch.Tensor] = []
        neumannBC_penalty: list[torch.Tensor] = []
        neumannBC_values: list[torch.Tensor] = []
        neumannBC_idx: list[torch.Tensor] = []

        for key in neumannBC:
            neumannBC_coordinates.append(
                torch.from_numpy(neumannBC[key]["coord"]).float().to(self.device)
            )
            neumannBC_coordinates[-1].requires_grad_(True)
            neumannBC_values.append(
                torch.from_numpy(neumannBC[key]["known_value"]).float().to(self.device)
            )
            neumannBC_penalty.append(
                torch.tensor(neumannBC[key]["penalty"]).float().to(self.device)
            )
            neumannBC_idx.append(
                torch.from_numpy(neumannBC[key]["idx"]).float().to(self.device)
            )

        optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.nn_parameters.learning_rate,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        strain_energy = StrainEnergy(
            self.to_parameters.E, self.to_parameters.nu, domain.dxdy
        )
        start_time = time.time()

        def closure_generator(t: int):
            def closure():
                u_pred = self.get_U(x, domain.length)
                u_pred.double()

                # ---- Calculate internal and external energies------
                internal_energy = strain_energy.calculate_objective(
                    u_pred, domain.shape, density
                )
                external_E = calculate_external_energy(
                    u_pred,
                    domain.dxdy,
                    neumannBC_idx,
                    neumannBC_values,
                )
                loss = internal_energy - external_E
                optimizer_LBFGS.zero_grad()
                loss.backward()

                if self.to_parameters.verbose:
                    print(
                        f"Iter: {t+1:d} Loss: {loss.item():.6e} "
                        + f"IntE: {internal_energy.item():.4e} ExtE: {external_E.item():.4e}"
                    )
                self.loss_array.append(loss.data.cpu())

                return float(loss)

            return closure

        for t in range(self.nn_parameters.iteration_count):
            # Zero gradients, perform a backward pass, and update the weights.

            optimizer_LBFGS.step(closure_generator(t))

            # Check convergence
            if self.convergence_check(
                self.loss_array, self.to_parameters.convergence_tolerances[iteration]
            ):
                break

        elapsed = time.time() - start_time
        training_times.append(elapsed)

        u_pred = self.get_U(x, domain.length)
        dfdrho, compliance = strain_energy.calculate_objective_gradient(
            u_pred, domain.shape, density
        )
        objective_values.append(compliance.cpu().detach().numpy())

        return dfdrho, compliance, elapsed

    def convergence_check(self, loss_array: list[float], tolerance: float):
        num_check = 10

        # Run minimum of 2*num_check iterations
        if len(loss_array) < 2 * num_check:
            return False

        mean1 = np.mean(loss_array[-2 * num_check : -num_check])
        mean2 = np.mean(loss_array[-num_check:])

        if np.abs(mean2) < 1e-6:
            return True

        if (np.abs(mean1 - mean2) / np.abs(mean2)) < tolerance:
            return True

        return False

    def get_U(self, x: torch.Tensor, length: float):
        u = self.model(x)
        phix = x[:, 0] / length

        if self.example == 1:
            Ux = phix * (1.0 - phix) * u[:, 0]
            Uy = phix * (1.0 - phix) * u[:, 1]
        elif self.example == 2:
            Ux = phix * u[:, 0]
            Uy = phix * u[:, 1]
        else:
            raise ValueError(f"Got unknown example {self.example}")

        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy), -1)
        return u_pred

    def evaluate_model(
        self, domain: Domain, density: npt.NDArray[np.float64], iteration: int
    ):
        Nx, Ny = domain.shape
        # why do I here need to tile x and repeat y, while train_model
        # uses a repeated x and tiled y?
        xy = np.array(
            [
                np.tile(domain.x_ray, len(domain.y_ray)),
                np.repeat(domain.y_ray, len(domain.x_ray)),
            ]
        ).T

        xy_tensor = torch.from_numpy(xy).float()
        xy_tensor = xy_tensor.to(self.device)
        xy_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xy_tensor)
        u_pred_torch = self.get_U(xy_tensor, domain.length)
        duxdxy = grad(
            u_pred_torch[:, 0].unsqueeze(1),
            xy_tensor,
            torch.ones(xy_tensor.size()[0], 1, device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        duydxy = grad(
            u_pred_torch[:, 1].unsqueeze(1),
            xy_tensor,
            torch.ones(xy_tensor.size()[0], 1, device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        E11 = duxdxy[:, 0].unsqueeze(1)
        E22 = duydxy[:, 1].unsqueeze(1)
        E12 = (duxdxy[:, 1].unsqueeze(1) + duydxy[:, 0].unsqueeze(1)) / 2

        E, nu = self.to_parameters.E, self.to_parameters.nu
        S11 = E / (1 - nu**2) * (E11 + nu * E22)
        S22 = E / (1 - nu**2) * (E22 + nu * E11)
        S12 = E * E12 / (1 + nu)

        u_pred = u_pred_torch.detach().cpu().numpy()

        # flag = np.ones( [Nx*Ny,1] )
        # threshold = np.max( density ) * 0.6
        # mask = ( density.flatten() < threshold )
        # flag[ mask , 0 ] = np.nan

        E11_pred = E11.detach().cpu().numpy()  # * flag
        E12_pred = E12.detach().cpu().numpy()  # * flag
        E22_pred = E22.detach().cpu().numpy()  # * flag
        S11_pred = S11.detach().cpu().numpy()  # * flag
        S12_pred = S12.detach().cpu().numpy()  # * flag
        S22_pred = S22.detach().cpu().numpy()  # * flag

        # u_pred[:,0] *= flag[:,0]
        # u_pred[:,1] *= flag[:,0]
        surUx = u_pred[:, 0].reshape(Ny, Nx)
        surUy = u_pred[:, 1].reshape(Ny, Nx)
        surUz = np.zeros([Nx, Ny])

        E11 = E11_pred.reshape(Ny, Nx)
        E12 = E12_pred.reshape(Ny, Nx)
        E22 = E22_pred.reshape(Ny, Nx)
        S11 = S11_pred.reshape(Ny, Nx)
        S12 = S12_pred.reshape(Ny, Nx)
        S22 = S22_pred.reshape(Ny, Nx)

        SVonMises = np.float64(
            np.sqrt(
                0.5 * ((S11 - S22) ** 2 + (S22) ** 2 + (-S11) ** 2 + 6 * (S12**2))
            )
        )
        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))

        np.save(
            f"{self.to_parameters.output_folder}/itr_{iteration}.npy",
            np.array(
                [density, U, S11, S12, S22, E11, E12, E22, SVonMises], dtype=object
            ),
        )

        write_fig = False
        if write_fig:
            _, axs = plt.subplots(3, 3, sharex="col")
            data_dict = {
                "Ux": (axs[0, 0], U[0]),
                "Uy": (axs[0, 1], U[1]),
                "Mises stress": (axs[0, 2], SVonMises),
                "E11": (axs[1, 0], E11),
                "E22": (axs[1, 1], E22),
                "E12": (axs[1, 2], E12),
                "S11": (axs[2, 0], S11),
                "S22": (axs[2, 1], S22),
                "S12": (axs[2, 2], S12),
            }

            for name, (ax, data) in data_dict.items():
                ff = ax.imshow(np.flipud(data), extent=domain.extent, cmap="jet")
                plt.colorbar(ff, ax=ax)
                ax.set_aspect("equal")
                ax.set_title(name)
            plt.tight_layout()

            util.smart_savefig(
                f"{self.to_parameters.output_folder}/field_vars_{iteration}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()
