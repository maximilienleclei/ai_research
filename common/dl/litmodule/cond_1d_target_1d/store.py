from autoregression import (
    Cond1DTarget1DAutoregressionLitModule,
    Cond1DTarget1DAutoregressionLitModuleConfig,
)
from diffusion import (
    Cond1DTarget1DDiffusionLitModule,
    Cond1DTarget1DDiffusionLitModuleConfig,
)
from hydra_zen import ZenStore

from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store(
        generate_config(
            Cond1DTarget1DAutoregressionLitModule,
            config=Cond1DTarget1DAutoregressionLitModuleConfig(),
        ),
        name="cond1dtarget1d/autoregression",
        group="litmodule",
    )
    store(
        generate_config(
            Cond1DTarget1DDiffusionLitModule,
            config=Cond1DTarget1DDiffusionLitModuleConfig(),
        ),
        name="cond1dtarget1d/diffusion",
        group="litmodule",
    )
