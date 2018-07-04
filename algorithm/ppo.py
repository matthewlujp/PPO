import os
import copy
from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim


class UpdateWithNoSamples(Exception):
    pass


class PPO:
    POLICY_PARAMS_FILE_NAME = "model.pth"
    VALUE_FN_PARAMS_FILE_NAME = "value_fn.pth"

    def __init__(self,
            policy: nn.Module,      # action = policy(observation)
            value_fn: nn.Module,    # value = value_fn(observation)
            update_epochs=10,       # epochs for updating policy using sampled data
            batch_size=32,          # batch_size <= num_actors x steps
            gamma=0.99,             # reward discount
            lamb=0.95,
            c_value_loss=1.0,       # coefficient for value function loss in the objective
            c_entropy_bonus=0.01,   # coefficient for entropy bonus in the objective
            clip_epsilon=0.2,       # epsilon used in the clipped surrogate objective
            learning_rate=0.0005      # learning rate
    ):
        self.__policy = policy
        self.__sampling_policy = copy.deepcopy(policy) # policy for sampling which does not track operations for autograd
        self.__sampling_policy.requires_grad_(False)

        self.__value_fn = value_fn    

        self.__hyper_params = {
            "update_epochs": update_epochs,
            "batch_size": batch_size,
            "gamma": gamma,
            "lambda": lamb,
            "c_value_fn_loss": c_value_loss,
            "c_entropy_bonus": c_entropy_bonus,
            "clip_epsilon": clip_epsilon,
            "learning_rate": learning_rate,
        }

        # internal sample buffer
        self.__samples = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "advantages": [],
            "target_state_values": [],
        }

    @property
    def policy(self) -> nn.Module:
        return self.__policy

    def reset_sample_buffers(self):
        ### Delete all samples in the internal buffer
        ### Call this method before feeding samples from trajectories obtained by N independent agents
        for n in self.__samples.keys():
            self.__samples[n] = []

    def act(self, observations: np.ndarray) -> np.ndarray:
        ### Calculate action using a policy with current parameters without building autograd graph
        ### This method is mainly for test
        ### Parameters:
        ### observations.shape == (batch_size, observation_dim) or (observation_dim,)
        ### Return:
        ### action.shape == (batch_size, action_dim) or (action_dim,)
        return  self.__sampling_policy(torch.from_numpy(observations).float()).sample().numpy()

    def feed_trajectory_samples(self, observations: list, actions: list, rewards: list, is_terminated: bool):
        ### Register samples from a trajectory of T steps obtained by one agent into the internal sample buffer.
        ### Parameters:
        ### observations.shape = (T+1, observation_dim)
        ### actions.shape = (T, action_dim)
        ### rewards.shape = (T,)
        ### is_terminated: whether this series of steps was terminated by an environment
        ### Note:
        ### T = len(observations) - 1, T <= _T, because it may be terminated on a way
        ### when you feed observations, actions, and rewards, advantage functions and target state values are automatically calculated as well
        ### all samples are converted to tensors with dtype=torch.float32 (not torch.double or torch.float64)

        assert len(observations) - 1 == len(actions) == len(rewards)
        assert len(observations[0]) == self.__policy.observation_dim
        assert len(actions[0]) == self.__policy.action_dim
        
        # convert np.ndarrays to torch.Tensors
        t_observations = torch.Tensor(observations)
        t_actions = torch.Tensor(actions)
        t_rewards = torch.Tensor(rewards)

        # calculate advantage functions
        advantages = self._calculate_advantages(t_observations, t_actions) # (A_0, A_1, ..., A_{T-1})

        # calculate target state values
        target_state_values = self._calculate_target_state_values(t_observations[-1], t_rewards, is_terminated) # (V^{tar}_0, V^{tar}_1, ..., V^{tar}_{T-1})

        # push to sample buffer
        self.__samples["observations"].append(t_observations[:-1]) # the final element of T+1 elements should be discarded, because update uses pairs of (s_t, a_t, r_t, A_t, V^{tar}_t)
        self.__samples["actions"].append(t_actions)
        self.__samples["rewards"].append(t_rewards)
        self.__samples["advantages"].append(advantages)
        self.__samples["target_state_values"].append(target_state_values)

    def _calculate_advantages(self, observations: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        ### Calculate advantages from T steps serial observations and rewards
        ### Parameters:
        ### observations.shape = torch.Size([T+1, observation_dim])
        ### rewards.shape == torch.Size([T])
        ### Return:
        ### advantages.shape == torch.Size([T])
        ### note: T = len(observations) - 1, T <= _T, because it may be terminated on a way
        
        T = len(observations) - 1
        td_errors = self._calculate_td_errors(observations, rewards) # (td_0, td_1, ..., td_{T-1})

        advantages = torch.Tensor(T)        
        advantages[T-1] = td_errors[T-1]
        for i in range(T-2, -1, -1):
            # A_i = td_i + \gamma * \lambda * A_{i+1}
            advantages[i] = td_errors[i] + self.__hyper_params["gamma"] * self.__hyper_params["lambda"] * advantages[i+1]

        return advantages

    def _calculate_td_errors(self, observations, rewards) -> torch.Tensor:
        ### Calculate a series of TD errors
        ### Parameters:
        ### observations.shape == torch.Size([T+1, observation_dim])
        ### rewards.shape == torch.Size([T])
        ### Return:
        ### td_errors.shape == torch.Size([T]) (td_0, td_1, ..., td_{T-1})
        
        with torch.no_grad(): # no grads should be calculated from target state values
            # td_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
            return rewards + self.__hyper_params["gamma"] * self.__value_fn(observations[1:]) - self.__value_fn(observations[:-1])

    def _calculate_target_state_values(self, final_observation: torch.Tensor, rewards: torch.Tensor, is_terminated: bool) -> torch.Tensor:
        ### Calculate train data for the state value function
        ### Parameters:
        ### final_observation.shape == torch.Size([observation_dim]): the final state of a trajectory of T steps, i.e., s_{t} of trajectory (s_0, a_0, s_1, a_1, ..., s_{t}))
        ### rewards.shape == torch.Size([T])
        ### is_terminated: whether trajectory is terminated (because of any failure)
        ### Return:
        ### target_state_values.shape == torch.Size([T])
        ### Note:
        ### T = len(rewards), T <= _T, because it may be terminated on a way

        T = len(rewards)
        target_state_values = torch.Tensor(T)
        
        with torch.no_grad():
            v_final = 0 if is_terminated else self.__value_fn(final_observation) # if a trajectory is terminated the final state value is assumed to be zero
        target_state_values[T-1] = rewards[T-1] + self.__hyper_params["gamma"] * v_final
        for i in range(T-2, -1, -1):
            # V^{tar}_t = r_t + \gamma * V^{tar}_{t+1}
            target_state_values[i] = rewards[i] + self.__hyper_params["gamma"] * target_state_values[i+1]

        return target_state_values
            
    def copy_policy(self, instance_num: int) -> list:
        ### Copy the current policy for concurrent sampling by N independent agents
        ### Parameters:
        ### instance_num: number of copies
        ### Return:
        ### a list of copied policies
        ### Note:
        ### the polices copied do not create calculation graph for autograd
        copied_policies = []
        for _ in range(instance_num):
            c_policy = copy.deepcopy(self.__sampling_policy)
            copied_policies.append(c_policy)
        return copied_policies
        
    def update(self) -> list:
        ### Update policy and state value function parameters using samples stored in the sample buffer.
        ### Return:
        ### list of averaged losses for each update epoch
        ### Note:
        ### if the sample buffer is empty, raise UpdateWithNoSamples exception
        ### feed_samples method should be called before this method is invoked
        
        for n, v in self.__samples.items():
            if len(v) == 0:
                raise UpdateWithNoSamples("internal sample buffer for {} is empty".format(n))

        # concatenate a list of tensors in sample buffers
        t_observations = torch.cat(self.__samples["observations"], 0)
        t_actions = torch.cat(self.__samples["actions"], 0)
        t_rewards = torch.cat(self.__samples["rewards"], 0)
        t_advantages = torch.cat(self.__samples["advantages"], 0)
        t_target_state_values = torch.cat(self.__samples["target_state_values"], 0)

        dataset = torch.utils.data.TensorDataset(t_observations, t_actions, t_rewards, t_advantages, t_target_state_values)
        optimizer = torch.optim.Adam(chain(self.__policy.parameters(), self.__value_fn.parameters()), lr=self.__hyper_params["learning_rate"])
        epoch_losses = []

        # update for update_epochs
        for e in range(self.__hyper_params["update_epochs"]):
            past_losses = torch.Tensor(int(np.ceil(len(dataset) / self.__hyper_params["batch_size"])))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.__hyper_params["batch_size"], shuffle=True)
            # batch training
            for i, (observations_, actions_, _, advantages_, target_state_values_) in enumerate(dataloader):
                optimizer.zero_grad()
                objective = self._calculate_objective(observations_, actions_, advantages_, target_state_values_)
                objective.backward(retain_graph=True)
                optimizer.step()
                past_losses[i] = float(objective.data) # save loss

            epoch_losses.append(float(past_losses.mean()))
        
        # update old policy
        self.__sampling_policy = copy.deepcopy(self.__policy)
        self.__sampling_policy.requires_grad_(False)

        # delete samples in the buffer
        self.reset_sample_buffers()

        return epoch_losses

    def _calculate_objective(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            advantages: torch.Tensor,
            target_state_values: torch.Tensor) -> torch.Tensor:
        ### Calculate surrogate objective for PPO and track operations
        ### Parameters:
        ### observations.shape == torch.Size([batch_size, observation_dim])
        ### actions.shape == torch.Size([batch_size, action_dim])
        ### rewards.shape == torch.Size([batch_size])
        ### advantages.shape == torch.Size([batch_size])
        ### target_state_values.shape == torch.Size([batch_size])
        ### Return:
        ### surrogate objective variable

        # L^{CLIP}
        ratios = self._calculate_ratios(observations, actions)
        unclamped_obj = ratios * advantages
        clamped_obj = torch.clamp(ratios, 1 - self.__hyper_params["clip_epsilon"], 1 + self.__hyper_params["clip_epsilon"]) * advantages
        # min{ratio * A, clip(ratios, 1-\epsilon, 1+\epsilon)}
        l_clip = torch.where(unclamped_obj < clamped_obj, unclamped_obj, clamped_obj).mean()

        # L^{VF}
        l_vf = nn.MSELoss()(self.__value_fn(observations).view(len(target_state_values)), target_state_values)

        # entropy bonus
        S = self.__policy(observations).entropy().sum(1).mean() # entropy of multivariate normal distribution is sum of element-wise entropies 
        
        objective = l_clip - self.__hyper_params["c_value_fn_loss"] * l_vf + self.__hyper_params["c_entropy_bonus"] * S
        return -objective # want to maximize

    def _calculate_ratios(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        ### Calculate weights for importance sampling
        ### Return:
        ### ratios.shape == torch.Size([batch_size])
        new_log_prob = self.__policy(observations).log_prob(actions).sum(1) # ln p(a) = ln p(a[0])p(a[1])...p(a[action_dim-1]) = ln p(a[0]) + ln p(a[1]) + ... + ln p(a[action_dim-1])
        old_log_prob = self.__sampling_policy(observations).log_prob(actions).sum(1) # old_policy is requires_grad = False
        return torch.exp(new_log_prob - old_log_prob) # exp(ln \frac(new_prob)(old_prob)) = exp(ln(new_prob) - ln(old_prob))

    def save(self, dir_path: str):
        ### Save current parameters of polic, value function, and other hyper parameters in a designated directory
        torch.save(self.__policy.state_dict(), os.path.join(dir_path, self.POLICY_PARAMS_FILE_NAME)) # save policy parameters
        torch.save(self.__policy.state_dict(), os.path.join(dir_path, self.VALUE_FN_PARAMS_FILE_NAME)) # save state value function parameters for further update

        # ToDo(matthewlujp): save other hyper parameters as well as current epochs

    def load(self, dir_path: str):
        ### Load saved parameters of policy, value function, and other hyper parameters
        # load policy
        policy_params_file_path = os.path.join(dir_path, self.POLICY_PARAMS_FILE_NAME)
        model_params = torch.load(policy_params_file_path)
        self.__policy.load_state_dict(model_params)

        # load value function
        value_fn_params_file_path = os.path.join(dir_path, self.VALUE_FN_PARAMS_FILE_NAME)
        value_fn_params = torch.load(value_fn_params)
        self.__value_fn.load_state_dict(value_fn_params)

        # ToDo(matthewlujp): load other hyper parameters so that restart training seamlessly



