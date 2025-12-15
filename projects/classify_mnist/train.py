from hydra_zen import ZenStore
from common.dl.litmodule.classification import BaseClassificationLitModuleConfig
from common.dl.runner import DeepLearningTaskRunner
from utils.hydra_zen import generate_config

from .datamodule import MNISTDataModule, MNISTDataModuleConfig
from .litmodule import MNISTClassificationLitModule


class TaskRunner(DeepLearningTaskRunner):

    @classmethod
    def store_configs(cls: type["TaskRunner"], store: ZenStore) -> None:
        super().store_configs(store=store)
        store(
            generate_config(MNISTDataModule, config=MNISTDataModuleConfig()),
            name="mnist",
            group="datamodule",
        )
        store(
            generate_config(
                MNISTClassificationLitModule,
                config=BaseClassificationLitModuleConfig(num_classes=10),
            ),
            name="classify_mnist",
            group="litmodule",
        )


TaskRunner.run_task()
