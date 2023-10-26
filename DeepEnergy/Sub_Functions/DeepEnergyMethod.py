import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

from Sub_Functions.MultiLayerNet import MultiLayerNet
from Sub_Functions.InternalEnergy import InternalEnergy
from Sub_Functions.IntegrationFext import IntegrationFext
from Sub_Functions import Utility as util
from Sub_Functions.data_structs import Domain, TopOptParameters, NNParameters


class DeepEnergyMethod:
    # Instance attributes
    def __init__(
        self,
        device,
        domain: Domain,
        example: int,
        to_parameters: TopOptParameters,
        nn_parameters: NNParameters,
    ):
        # self.data = data
        self.model = MultiLayerNet(
            nn_parameters.input_size,
            nn_parameters.neuron_count,
            nn_parameters.output_size,
            nn_parameters.activation_function,
            nn_parameters.CNN_deviation,
            nn_parameters.rff_deviation,
            nn_parameters.layer_count,
        )
        self.model = self.model.to(device)

        self.lossArray = []
        self.device = device
        self.domain = domain
        self.to_parameters = to_parameters
        self.nn_parameters = nn_parameters
        self.example = example
        self.FextLoss = IntegrationFext(nn_parameters.input_size)
        self.InternalEnergy = InternalEnergy(
            self.to_parameters.E, self.to_parameters.nu
        )

    def train_model(
        self,
        rho,
        data,
        iteration,
        neumannBC,
        dirichletBC,
        training_times,
        objective_values,
    ):
        x = torch.from_numpy(data).float()
        x = x.to(self.device)
        x.requires_grad_(True)

        density = torch.from_numpy(rho).float()
        density = torch.reshape(density, [self.domain.Ny - 1, self.domain.Nx - 1]).to(
            self.device
        )

        # -------------------------------------------------------------------------------
        #                           Neumann BC
        # -------------------------------------------------------------------------------
        neuBC_coordinates = {}
        neuBC_penalty = {}
        neuBC_values = {}
        neuBC_idx = {}

        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = (
                torch.from_numpy(neumannBC[keyi]["coord"]).float().to(self.device)
            )
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = (
                torch.from_numpy(neumannBC[keyi]["known_value"]).float().to(self.device)
            )
            neuBC_penalty[i] = (
                torch.tensor(neumannBC[keyi]["penalty"]).float().to(self.device)
            )
            neuBC_idx[i] = (
                torch.from_numpy(neumannBC[keyi]["idx"]).float().to(self.device)
            )

        # ----------------------------------------------------------------------------------
        # Minimizing loss function (energy and boundary conditions)
        # ----------------------------------------------------------------------------------
        optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.nn_parameters.learning_rate,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )
        start_time = time.time()
        energy_loss_array = []
        loss_history = np.zeros(self.nn_parameters.iteration_count)
        Iter_No_Hist = []

        def closure_generator(t):
            def closure():
                u_pred = self.get_U(x)
                u_pred.double()

                # ---- Calculate internal and external energies------
                storedEnergy = self.InternalEnergy.Elastic2DGaussQuad(
                    u_pred, x, self.domain.dxdy, self.domain.shape, density, False
                )
                externalE = self.FextLoss.lossFextEnergy(
                    u_pred,
                    x,
                    neuBC_coordinates,
                    neuBC_values,
                    neuBC_idx,
                    self.domain.dxdy,
                )
                loss = storedEnergy - externalE
                optimizer_LBFGS.zero_grad()
                loss.backward()

                if self.to_parameters.verbose:
                    print(
                        f"Iter: {t+1:d} Loss: {loss.item():.6e} "
                        + f"IntE: {storedEnergy.item():.4e} ExtE: {externalE.item():.4e}"
                    )
                loss_history[t] = loss.data
                energy_loss_array.append(loss.data)
                Iter_No_Hist.append(t)
                self.lossArray.append(loss.data.cpu())
                return loss

            return closure

        for t in range(self.nn_parameters.iteration_count):
            # Zero gradients, perform a backward pass, and update the weights.

            optimizer_LBFGS.step(closure_generator(t))

            # Check convergence
            if self.convergence_check(
                self.lossArray, self.to_parameters.convergence_tolerances[iteration]
            ):
                break

        elapsed = time.time() - start_time
        print(f"Training time: {elapsed:.4f}")
        training_times.append(elapsed)

        u_pred = self.get_U(x)
        dfdrho, compliance = self.InternalEnergy.Elastic2DGaussQuad(
            u_pred, x, self.domain.dxdy, self.domain.shape, density, True
        )
        objective_values.append(compliance.cpu().detach().numpy())

        return dfdrho, compliance

    def convergence_check(self, arry, rel_tol):
        num_check = 10

        # Run minimum of 2*num_check iterations
        if len(arry) < 2 * num_check:
            return False

        mean1 = np.mean(arry[-2 * num_check : -num_check])
        mean2 = np.mean(arry[-num_check:])

        if np.abs(mean2) < 1e-6:
            print("Loss value converged to abs tol of 1e-6")
            return True

        if (np.abs(mean1 - mean2) / np.abs(mean2)) < rel_tol:
            print("Loss value converged to rel tol of " + str(rel_tol))
            return True

        return False

    def get_U(self, x):
        u = self.model(
            x, self.nn_parameters.layer_count, self.nn_parameters.activation_function
        )
        phix = x[:, 0] / self.domain.length

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

    def evaluate_model(self, x, y, density, iteration):
        Nx = len(x)
        Ny = len(y)
        xGrid, yGrid = np.meshgrid(x, y)
        x1D = xGrid.flatten()
        y1D = yGrid.flatten()
        xy = np.concatenate((np.array([x1D]).T, np.array([y1D]).T), axis=-1)
        xy_tensor = torch.from_numpy(xy).float()
        xy_tensor = xy_tensor.to(self.device)
        xy_tensor.requires_grad_(True)
        # u_pred_torch = self.model(xy_tensor)
        u_pred_torch = self.get_U(xy_tensor)
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

        # Write output
        Write_file = True
        if Write_file:
            z = np.array([0]).astype(np.float64)
            util.write_vtk_v2(
                f"{self.to_parameters.output_folder}/itr_{iteration}",
                x,
                y,
                z,
                U,
                S11,
                S12,
                S22,
                E11,
                E12,
                E22,
                SVonMises,
            )

        Write_fig = True
        if Write_fig:
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
                ff = ax.imshow(np.flipud(data), extent=self.domain.extent, cmap="jet")
                plt.colorbar(ff, ax=ax, shrink=0.5)
                ax.set_aspect("equal")
                ax.set_title(name)
            plt.tight_layout()

            util.smart_savefig(
                f"{self.to_parameters.output_folder}/field_vars_{iteration}.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()

        np.save(
            f"{self.to_parameters.output_folder}/itr_{iteration}.npy",
            np.array(
                [density, U, S11, S12, S22, E11, E12, E22, SVonMises], dtype=object
            ),
        )
