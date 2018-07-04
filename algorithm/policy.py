import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions.normal as normal


class MLPGaussianPolicy(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_layers=None, # list
            action_high=None,   # np.ndarray
            action_low=None,    # np.ndarray
            activation_fn=F.relu):
        super().__init__()
        self.__observation_dim = observation_dim
        self.__action_dim = action_dim
        self.__action_high = action_high
        self.__action_low = action_low
        self.__activation_fn = activation_fn

        layers = []
        l_dim = observation_dim
        if hidden_layers is not None:
            for hl in hidden_layers:
                layers.append(nn.Linear(l_dim, hl))
                l_dim = hl
        layers.append(nn.Linear(l_dim, action_dim))
        self.__layers = nn.ModuleList(layers)

        # variables for standard deviations
        std = 1./ np.sqrt(action_dim)
        self.__log_stddev = Parameter(torch.Tensor(action_dim).data.uniform_(0.01, std))

    @property
    def observation_dim(self) -> int:
        return self.__observation_dim

    @property
    def action_dim(self) -> int:
        return self.__action_dim

    def requires_grad_(self, requires: bool):
        ### If set to False, operations are not tracked
        for p in self.parameters():
            p.requires_grad_(requires)

    def forward(self, x: torch.Tensor) -> normal.Normal:
        # x.shape == torch.Size(batch_size, observation_dim)
        # -> action.shape == torch.Size(batch_size, action_dim)
        for l in self.__layers[:-1]:
            x = self.__activation_fn(l(x))
        x = self.__layers[-1](x)

        if self.__action_high is None or self.__action_low is None: # case where either of these is none will not happen
            return normal.Normal(loc=x, scale=torch.exp(self.__log_stddev))
        else:
            action_high = torch.from_numpy(self.__action_high).float()
            action_low = torch.from_numpy(self.__action_low).float()
            limitted_x = F.tanh(x) / 2.0 * (action_high - action_low) + action_low # may contains nan
            means = torch.where(
                (action_high == float('inf')) + (action_low == -float('inf')),
                x, 
                limitted_x,
            )
            return normal.Normal(
                loc=means,
                scale=torch.exp(self.__log_stddev),
            )

    def act(self, observations: np.ndarray, deterministic=False) -> np.ndarray:
        ### Calculate action based on observation
        ### This method is for raw usage of an instance of this class without importing PyTorch
        ### Parameters:
        ### observations.shape == (batch_size, observation_dim) or (observation_dim,)
        ### deterministic: if True, the policy returns a deterministic action, i.e, not sampling from a multivariable normal distribution
        ### Return:
        ### action.shape == (batch_size, action_dim) or (action_dim,)
        t_observations = torch.from_numpy(observations).float()
        with torch.no_grad():
            distribution = self.forward(t_observations)
            if deterministic:
                return distribution.mean.numpy()
            else:
                return distribution.sample().numpy()