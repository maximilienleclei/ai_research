""":class:`.CDAM`.

---

Shapes:
    - BS: Batch size
    - SL: Sequence length
    - SLM1: Sequence length minus 1
    - NTF: Number of target features
        (:paramref:`.BaseCAMConfig.num_target_features`)
    - NCF: Number of conditioning features
        (:paramref:`.BaseCAMConfig.num_conditioning_features`)
    - NIF: Number of input features
        (:attr:`.BaseCAM.num_input_features`)
    - NOL: Number of output logits
        (:attr:`.BaseCAM.num_output_logits`)
"""

import logging

import torch.nn.functional as f
from jaxtyping import Float
from torch import Tensor

from .base import BaseCAM

log = logging.getLogger(__name__)


class CDAM(BaseCAM):
    """A Conditional Discriminative Autoregressive Model."""

    def compute_loss(
        self: "CDAM",
        predicted_logits: Float[Tensor, " BS SL NOL"],
        target_features: Float[Tensor, " BS SL NTF"],
    ) -> Float[Tensor, " "]:
        return f.mse_loss(predicted_logits, target_features)

    def predict_sequence_logits_and_features_fitting(
        self: "CDAM",
        conditioning_sequence_features: Float[Tensor, " BS SL NCF"],
        target_sequence_features: Float[Tensor, " BS SL NTF"],
    ) -> tuple[Float[Tensor, " BS SL NOL"], Float[Tensor, " BS SL NTF"]]:
        predicted_sequence_logits = predicted_sequence_features = (
            self.predict_sequence_logits_fitting(
                conditioning_sequence_features,
                target_sequence_features,
            )
        )
        return predicted_sequence_logits, predicted_sequence_features

    def predict_timestep_features_inference(
        self: "CDAM",
        predicted_timestep_logits: Float[Tensor, " BS 1 NOL"],
    ) -> Float[Tensor, " BS 1 NTF"]:
        return predicted_timestep_logits
