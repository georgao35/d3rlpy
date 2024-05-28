import dataclasses

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_continuous_q_function,
    create_normal_policy,
    create_value_function,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.optimizers import OptimizerFactory, make_optimizer_field
from ...models.q_functions import MeanQFunctionFactory
from ...types import Shape
from .base import QLearningAlgoBase
from .torch.iql_impl import IQLModules
from .torch.xql_impl import XQLImpl

__all__ = ["XQLConfig", "XQL"]


@dataclasses.dataclass()
class XQLConfig(LearnableConfig):
    r"""Extreme Q-Learning algorithm."""

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    value_encoder_factory: EncoderFactory = make_encoder_field()
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    expectile: float = 0.7
    weight_temp: float = 3.0
    max_weight: float = 100.0
    beta: float = 1.0
    max_clip: float = 7.0
    use_log_loss: bool = False

    def create(self, device: DeviceArg = False) -> "XQL":
        return XQL(self, device)

    @staticmethod
    def get_type() -> str:
        return "xql"


class XQL(QLearningAlgoBase[XQLImpl, XQLConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        policy = create_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
            device=self._device,
        )
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            MeanQFunctionFactory(),
            n_ensembles=self._config.n_critics,
            device=self._device,
        )
        value_func = create_value_function(
            observation_shape,
            self._config.value_encoder_factory,
            device=self._device,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(), lr=self._config.actor_learning_rate
        )
        q_func_params = list(q_funcs.named_modules())
        v_func_params = list(value_func.named_modules())
        critic_optim = self._config.critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._config.critic_learning_rate
        )

        modules = IQLModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            value_func=value_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
        )

        self._impl = XQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            expectile=self._config.expectile,
            weight_temp=self._config.weight_temp,
            max_weight=self._config.max_weight,
            device=self._device,
            beta=self._config.beta,
            max_clip=self._config.max_clip,
            use_log_loss=self._config.use_log_loss,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(XQLConfig)
