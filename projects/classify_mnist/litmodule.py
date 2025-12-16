import torch
import wandb

from common.dl.litmodule.classification import BaseClassificationLitModule


class MNISTClassificationLitModule(BaseClassificationLitModule):

    @property
    def wandb_media_x(self):
        def convert(x):
            # Unnormalize: original = (normalized * std) + mean
            x = x * 0.3081 + 0.1307
            # Convert to 0-255 uint8
            x = (x.clamp(0, 1) * 255).to(torch.uint8)
            return wandb.Image(x)
        return convert
