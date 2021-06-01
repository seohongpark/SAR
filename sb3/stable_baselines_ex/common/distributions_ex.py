from typing import Tuple, Optional

import torch as th
from torch import nn
from torch.distributions import Normal

from stable_baselines3.common.distributions import DiagGaussianDistribution, sum_independent_dims, Distribution, \
    TanhBijector, CategoricalDistribution
from stable_baselines3.common.preprocessing import get_action_dim


class Compound(nn.Module):
    def __init__(self, in_features: int, out_features: int, init, spec) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spec = spec
        self.linears = nn.ModuleList([])
        self.params = nn.ParameterList([])
        for type_, dim in self.spec:
            if dim == 0:
                continue
            if type_ == 'linear':
                self.linears.append(nn.Linear(in_features, dim))
            else:
                self.params.append(nn.Parameter(th.ones(dim) * init, requires_grad=True))

    def forward(self, x):
        outputs = []
        linear_idx = 0
        param_idx = 0
        for type_, dim in self.spec:
            if dim == 0:
                continue
            if type_ == 'linear':
                linear = self.linears[linear_idx]
                linear_idx += 1
                outputs.append(linear(x))
            else:
                param = self.params[param_idx]
                param_idx += 1
                outputs.append(param.unsqueeze(0).expand(x.size(0), -1))
        return th.cat(outputs, dim=1)


class DiagGaussianDistributionEx(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim, mean_type='full', std_type='shared', ori_action_dim=None):
        super().__init__()
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

        self.mean_type = mean_type
        self.std_type = std_type
        self.ori_action_dim = ori_action_dim

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0, mean_init=0.0) -> Tuple[nn.Module, nn.Module]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :return:
        """
        if self.mean_type == 'full':
            mean_actions = nn.Linear(latent_dim, self.action_dim)
        elif self.mean_type == 'mixed':
            mean_actions = Compound(
                latent_dim, self.action_dim, init=mean_init,
                spec=[('linear', self.ori_action_dim), ('param', self.action_dim - self.ori_action_dim)],
            )
        else:
            raise Exception('Unknown mean_type')

        if self.std_type == 'full':
            log_std_actions = nn.Linear(latent_dim, self.action_dim)
        elif self.std_type == 'mixed':
            log_std_actions = Compound(
                latent_dim, self.action_dim, init=log_std_init,
                spec=[('param', self.ori_action_dim), ('linear', self.action_dim - self.ori_action_dim)],
            )
        elif self.std_type == 'shared':
            log_std_actions = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        else:
            raise Exception('Unknown std_type')

        return mean_actions, log_std_actions

    def proba_distribution(self, mean_actions: th.Tensor, log_std_actions: th.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        if self.std_type in ['full', 'mixed']:
            action_std = log_std_actions.exp()
        else:
            action_std = th.ones_like(mean_actions) * log_std_actions.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std_actions: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std_actions)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std_actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std_actions)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SquashedDiagGaussianDistributionEx(DiagGaussianDistributionEx):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6, **kwargs):
        super().__init__(action_dim, **kwargs)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "SquashedDiagGaussianDistributionEx":
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - actions ** 2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return th.tanh(self.gaussian_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return th.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


class MixedDistribution(Distribution):
    def __init__(self, cont_cls, cont_kwargs, action_space):
        super().__init__()
        self.cont_dist = cont_cls(get_action_dim(action_space.spaces[0]), **cont_kwargs)
        self.dis_dist = CategoricalDistribution(action_space.spaces[1].n)

    def proba_distribution_net(self, latent_dim: int, log_std_init):
        action_net, log_std_net = self.cont_dist.proba_distribution_net(latent_dim, log_std_init)
        dist_action_net = self.dis_dist.proba_distribution_net(latent_dim)
        return action_net, log_std_net, dist_action_net

    def proba_distribution(self, mean_actions: th.Tensor, log_std_actions: th.Tensor, action_logits) -> "MixedDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        self.distribution = (
            self.cont_dist.proba_distribution(mean_actions, log_std_actions),
            self.dis_dist.proba_distribution(action_logits),
        )
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        return self.distribution[0].log_prob(actions[:, :-1]) + self.distribution[1].log_prob(actions[:, -1])

    def entropy(self) -> th.Tensor:
        return self.distribution[0].entropy() + self.distribution[1].entropy()

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return th.cat([self.distribution[0].sample(), self.distribution[1].sample().unsqueeze(0)], dim=1)

    def mode(self) -> th.Tensor:
        return th.cat([self.distribution[0].mode(), self.distribution[1].mode().unsqueeze(0)], dim=1)

    def actions_from_params(self, mean_actions: th.Tensor, log_std_actions: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        raise NotImplementedError()
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std_actions)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std_actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        raise NotImplementedError()
        actions = self.actions_from_params(mean_actions, log_std_actions)
        log_prob = self.log_prob(actions)
        return actions, log_prob
