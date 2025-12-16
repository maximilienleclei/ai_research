from hydra_zen import ZenStore

from common.dl.litmodule.cond1d_target1d.autoregression import (
    Cond1DTarget1DAutoregressionLitModule,
    Cond1DTarget1DAutoregressionLitModuleConfig,
)
from common.dl.litmodule.cond1d_target1d.diffusion import (
    Cond1DTarget1DDiffusionLitModule,
    Cond1DTarget1DDiffusionLitModuleConfig,
)
from common.utils.hydra_zen import generate_config


def store_configs(store: ZenStore) -> None:
    store(
        generate_config(
            Cond1DTarget1DAutoregressionLitModule,
            config=generate_config(
                Cond1DTarget1DAutoregressionLitModuleConfig
            ),
        ),
        name="cond1dtarget1d/autoregression",
        group="litmodule",
    )
    store(
        generate_config(
            Cond1DTarget1DDiffusionLitModule,
            config=generate_config(Cond1DTarget1DDiffusionLitModuleConfig),
        ),
        name="cond1dtarget1d/diffusion",
        group="litmodule",
    )
