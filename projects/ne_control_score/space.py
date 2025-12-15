from dataclasses import dataclass

from omegaconf import MISSING
from common.ne.space import BaseReinforcementSpace, BaseSpaceConfig
from torchrl.envs.libs.gym import GymEnv


@dataclass
class GymReinforcementSpaceConfig(BaseSpaceConfig):

    env_name: str = MISSING


class GymReinforcementSpace(BaseReinforcementSpace):

    def __init__(
        self: "GymReinforcementSpace",
        config: GymReinforcementSpaceConfig,
    ) -> None:
        super().__init__(config=config, env=GymEnv(env_name=config.env_name))
