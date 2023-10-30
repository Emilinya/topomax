import torch
import rff.layers

from Sub_Functions.data_structs import NNParameters


class MultiLayerNet(torch.nn.Module):
    def __init__(self, parameters: NNParameters):
        super().__init__()

        input_size = parameters.input_size
        output_size = parameters.output_size
        neuron_count = parameters.neuron_count
        rff_deviation = parameters.rff_deviation
        CNN_deviation = parameters.CNN_deviation

        self.layer_count = parameters.layer_count
        self.activation_function = getattr(torch, parameters.activation_function)
        self.encoding = rff.layers.GaussianEncoding(
            sigma=rff_deviation, input_size=input_size, encoded_size=neuron_count // 2
        )

        if self.layer_count < 2:
            raise ValueError(
                "Can't create MultiLayerNet with less than 2 layers "
                + f"(tried to create one with {self.layer_count} layers)"
            )

        self.linears = torch.nn.ModuleList()
        for i in range(self.layer_count):
            linear_inputs = input_size if i == 0 else neuron_count
            linear_outputs = output_size if i == self.layer_count - 1 else neuron_count
            self.linears.append(torch.nn.Linear(linear_inputs, linear_outputs))

            torch.nn.init.constant_(self.linears[i].bias, 0.0)
            torch.nn.init.normal_(self.linears[i].weight, mean=0, std=CNN_deviation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoding(x)
        for i in range(1, self.layer_count - 1):
            y = self.activation_function(self.linears[i](y))
        y = self.linears[-1](y)

        return y
