import dataclasses

import torch

from ....models.torch import ContinuousEnsembleQFunctionForwarder
from ....torch_utility import TorchMiniBatch
from ....types import Shape, TorchObservation
from .iql_impl import IQLCriticLoss, IQLImpl, IQLModules

__all__ = ["XQLImpl"]


class XQLImpl(IQLImpl):
    _modules: IQLModules
    _use_log_loss: bool
    _beta: float
    _max_clip: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: IQLModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        beta: float,
        max_clip: float,
        use_log_loss: bool,
        device: str,
    ):
        super().__init__(
            observation_shape,
            action_size,
            modules,
            q_func_forwarder,
            targ_q_func_forwarder,
            gamma,
            tau,
            expectile,
            weight_temp,
            max_weight,
            device,
        )
        self._beta = beta
        self._max_clip = max_clip
        self._use_log_loss = use_log_loss

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> IQLCriticLoss:
        q_loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

        v_t = self._modules.value_func(batch.observations)
        with torch.no_grad():
            q_t = self._targ_q_func_forwarder.compute_expected_q(
                batch.observations, batch.actions, "min"
            )
        if self._use_log_loss:
            value_loss = self.gumbel_log_loss(
                q_t - v_t, self._beta, self._max_clip
            )
        else:
            value_loss = self.gumbel_loss(q_t - v_t, self._beta, self._max_clip)

        return IQLCriticLoss(
            critic_loss=q_loss + value_loss, q_loss=q_loss, v_loss=value_loss
        )

    @staticmethod
    def gumbel_loss(diff: torch.Tensor, beta, clip) -> torch.Tensor:
        z = diff / beta
        z = torch.clamp(z, None, clip)
        max_z = torch.max(z, dim=0)[0]
        max_z = torch.where(max_z < -1.0, torch.tensor(-1.0), max_z)
        max_z = max_z.detach()
        loss = torch.exp(z - max_z) - z * torch.exp(-max_z) - torch.exp(-max_z)
        return loss.mean()

    @staticmethod
    def gumbel_log_loss(diff: torch.Tensor, beta, clip) -> torch.Tensor:
        z = diff / beta
        z = torch.clamp(z, None, clip)
        max_z = torch.max(z, dim=0)[0]
        max_z = torch.where(max_z < -1.0, torch.tensor(-1.0), max_z)

        z1 = z - max_z
        grad = (torch.exp(z1) - torch.exp(-max_z)) / (
            torch.exp(z1) - z * torch.exp(-max_z)
        ).mean(dim=0, keepdim=True)
        return grad.detach() * z
