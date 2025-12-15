import wandb
from common.dl.litmodule.classification import BaseClassificationLitModule


class MNISTClassificationLitModule(BaseClassificationLitModule):

    @property
    def wandb_media_x(self):  # type: ignore[no-untyped-def] # noqa: ANN201
        return wandb.Image
