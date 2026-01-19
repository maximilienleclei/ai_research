from hydra_zen import ZenStore
from torch.utils.data import DataLoader

from common.dl.litmodule.classification import (
    BaseClassificationLitModuleConfig,
)
from common.dl.runner import DeepLearningTaskRunner
from common.utils.hydra_zen import generate_config, generate_config_partial
from projects.dl_classify_mnist.datamodule import (
    MNISTDataModule,
    MNISTDataModuleConfig,
)
from projects.dl_classify_mnist.litmodule import MNISTClassificationLitModule


class TaskRunner(DeepLearningTaskRunner):

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store=store)
        store(
            generate_config(
                MNISTDataModule,
                config=generate_config(MNISTDataModuleConfig),
                dataloader=generate_config_partial(DataLoader),
            ),
            name="dl_classify_mnist",
            group="datamodule",
        )
        store(
            generate_config(
                MNISTClassificationLitModule,
                config=generate_config(BaseClassificationLitModuleConfig),
            ),
            name="dl_classify_mnist",
            group="litmodule",
        )


TaskRunner.run_task()
