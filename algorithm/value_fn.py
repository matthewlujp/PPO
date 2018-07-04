import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPValueFn(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            hidden_layers=None,
            activation_fn=F.relu):
        super().__init__()
        self.__observation_dim = observation_dim
        self.__activation_fn = activation_fn

        layers = []
        l_dim = observation_dim
        if hidden_layers is not None:
            for hl in hidden_layers:
                layers.append(nn.Linear(l_dim, hl))
                l_dim = hl
        layers.append(nn.Linear(l_dim, 1))
        self.__layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape == torch.Size([batch_size, observation_dim])
        # -> value.shape == torch.Size([batch_size,])
        for l in self.__layers[:-1]:
            x = self.__activation_fn(l(x))
        x = self.__layers[-1](x)
        return x