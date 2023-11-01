import torch
import numpy as np
import numpy.typing as npt

from DeepEnergy.src.StrainEnergy import StrainEnergy
from DeepEnergy.src.MultiLayerNet import MultiLayerNet
from DeepEnergy.src.data_structs import Domain, NNParameters
from DeepEnergy.src.external_energy import calculate_external_energy
from DeepEnergy.src.bc_helpers import TractionPoints, DirichletEnforcer


class DeepEnergyMethod:
    # Instance attributes
    def __init__(
        self,
        E: float,
        nu: float,
        device: torch.device,
        nn_parameters: NNParameters,
        dirichlet_enforcer: DirichletEnforcer,
        traction_points_list: list[TractionPoints],
    ):
        # self.data = data
        self.model = MultiLayerNet(nn_parameters)
        self.model = self.model.to(device)

        self.E = E
        self.nu = nu
        self.device = device
        self.nn_parameters = nn_parameters
        self.dirichlet_enforcer = dirichlet_enforcer
        self.traction_points_list = traction_points_list

        self.loss_array = []

    def train_model(self, rho: npt.NDArray[np.float64], domain: Domain):
        x = torch.from_numpy(domain.coordinates).float()
        x = x.to(self.device)
        x.requires_grad_(True)

        density = torch.from_numpy(rho).float()
        density = torch.reshape(density, [domain.Ny - 1, domain.Nx - 1]).to(self.device)

        neumannBC_values: list[torch.Tensor] = []
        neumannBC_idx: list[torch.Tensor] = []

        for traction_points in self.traction_points_list:
            neumannBC_values.append(
                torch.from_numpy(traction_points.known_value).float().to(self.device)
            )
            neumannBC_idx.append(
                torch.from_numpy(traction_points.idx).float().to(self.device)
            )

        optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.nn_parameters.learning_rate,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        strain_energy = StrainEnergy(self.E, self.nu, domain.dxdy)

        def closure_generator(t: int):
            def closure():
                u_pred = self.get_U(x, domain)
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

                if self.nn_parameters.verbose:
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
                self.loss_array,
                self.nn_parameters.convergence_tolerance,
            ):
                break

        u_pred = self.get_U(x, domain)
        dfdrho, compliance = strain_energy.calculate_objective_gradient(
            u_pred, domain.shape, density
        )

        return compliance, dfdrho

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

    def get_U(
        self,
        x: torch.Tensor,
        domain: Domain,
    ):
        u_tilde: torch.Tensor = self.model(x)

        normed_x = (x[:, 0] / domain.length).unsqueeze(1)
        normed_y = (x[:, 1] / domain.height).unsqueeze(1)

        return self.dirichlet_enforcer(u_tilde, normed_x, normed_y)
