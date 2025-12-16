"""Shapes:

- BS: Batch size
- SL: Sequence length
- SLM1: Sequence length minus 1
- NTF: Number of target features
- NCF: Number of conditioning features
- NIF: Number of input features
- NOL: Number of output logits
- HS: Hidden size
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Annotated as An

import torch
from einops import rearrange
from feedforward import FNN, FNNConfig
from jaxtyping import Float
from mamba2 import Mamba2
from mambapy.mamba import Mamba, MambaConfig
from mambapy.mamba2 import Mamba2Config
from torch import Tensor, nn
from torch.nn import LSTM, RNN
from utils.beartype import ge, one_of

log = logging.getLogger(__name__)


@dataclass
class BaseCAMConfig:
    """Args:

    num_conditioning_features: Number of features used to condition
        the prediction at each time step.
    num_target_features: Number of features to be predicted at each
        time step.
    teacher_forcing: Whether to feed the true target features from
        the previous time step to the model during training (and its
        own predictions during inference).
    """

    num_conditioning_features: An[int, ge(0)] = 513
    num_target_features: An[int, ge(1)] = 4
    teacher_forcing: bool = True


class BaseCAM(nn.Module, ABC):
    """Base Conditional Autoregressive Model.

    During training: conditioning features are concatenated with
    target features from the previous time step (if `teacher_forcing`,
    otherwise zeroes) and passes them through the model to predict the
    target features for the current time step.

    During inference: instead of using the true target features from the
    previous time step, the model is fed its own predictions from the
    previous time step (if `teacher_forcing`, otherwise
    zeroes once again).
    """

    def __init__(
        self: "BaseCAM",
        config: BaseCAMConfig,
        model_partial: partial[
            FNNConfig | RNN | LSTM | MambaConfig | Mamba2Config
        ],
    ) -> None:
        super().__init__()
        self.config = config
        self.num_input_features = (
            config.num_conditioning_features + config.num_target_features
        )
        if not hasattr(self, "num_output_logits"):
            self.num_output_logits = config.num_target_features

        if model_partial.func == FNNConfig:
            if "input_size" in model_partial.keywords:
                log.warning(
                    "`input_size` overwritten by the sum of "
                    "`num_conditioning_features` and "
                    "`num_target_features`.",
                )
            if "output_size" in model_partial.keywords:
                log.warning(
                    "`output_size` overwritten by `num_output_logits`.",
                )
            self.model = FNN(
                model_partial(
                    input_size=self.num_input_features,
                    output_size=self.num_output_logits,
                ),
            )
        elif model_partial.func in [RNN, LSTM]:
            log.warning(
                "`input_size` overwritten by the sum of "
                "`num_conditioning_features` and `num_target_features`.",
            )
            hidden_size = model_partial.keywords.get("hidden_size", None)
            self.model: RNN | LSTM = model_partial(
                input_size=self.num_input_features,
                batch_first=True,
            )
            self.proj_out = nn.Linear(
                in_features=hidden_size,
                out_features=self.num_output_logits,
            )
        else:
            if model_partial.func == MambaConfig:
                self.model: Mamba = Mamba(model_partial())
            else:  # model_partial.func == Mamba2Config:
                self.model: Mamba2 = Mamba2(model_partial())
            self.proj_in = nn.Linear(
                in_features=self.num_input_features,
                out_features=self.model.config.d_model,
            )
            self.proj_out = nn.Linear(
                in_features=self.model.config.d_model,
                out_features=self.num_output_logits,
            )

    def predict_sequence_features_and_compute_loss(
        self: "BaseCAM",
        conditioning_sequence_features: Float[Tensor, " BS SL NCF"],
        target_sequence_features: Float[Tensor, " BS SL NTF"],
        mode: An[str, one_of("fitting", "inference")],
    ) -> tuple[Float[Tensor, " BS SL NTF"], Float[Tensor, " "]]:
        predicted_sequence_logits, predicted_sequence_features = (
            self.predict_sequence_logits_and_features_fitting(
                conditioning_sequence_features,
                target_sequence_features,
            )
            if mode == "fitting"
            else self.predict_sequence_logits_and_features_inference(
                conditioning_sequence_features,
            )
        )
        loss: Float[Tensor, " "] = self.compute_loss(
            predicted_sequence_logits,
            target_sequence_features,
        )
        return predicted_sequence_features, loss

    @abstractmethod
    def compute_loss(
        self: "BaseCAM",
        predicted_sequence_logits: Float[Tensor, " BS SL NOL"],
        target_sequence_features: Float[Tensor, " BS SL NTF"],
    ) -> Float[Tensor, " "]: ...

    @abstractmethod
    def predict_sequence_logits_and_features_fitting(
        self: "BaseCAM",
        conditioning_sequence_features: Float[Tensor, " BS SL NCF"],
        target_sequence_features: Float[Tensor, " BS SL NTF"],
    ) -> tuple[Float[Tensor, " BS SL NOL"], Float[Tensor, " BS SL NTF"]]: ...

    def predict_sequence_logits_fitting(
        self: "BaseCAM",
        conditioning_sequence_features: Float[Tensor, " BS SL NCF"],
        target_sequence_features: Float[Tensor, " BS SL NTF"],
    ) -> Float[Tensor, " BS SL NOL"]:
        """`conditioning_sequence_features` is concatenated with
        `target_sequence_features` (if `teacher_forcing`) or a
        zeroed-out tensor of `target_sequence_features`'s shape before
        being passed through the model.
        """
        if self.config.teacher_forcing:
            single_timestep_zeroed_target_features: Float[
                Tensor,
                " BS 1 NTF",
            ] = torch.zeros_like(target_sequence_features[:, :1])
            all_but_last_timestep_target_features: Float[
                Tensor,
                " BS SLM1 NTF",
            ] = target_sequence_features[:, :-1]
            shifted_sequence_target_features: Float[Tensor, " BS SL NTF"] = (
                torch.cat(
                    (
                        single_timestep_zeroed_target_features,
                        all_but_last_timestep_target_features,
                    ),
                    dim=1,
                )
            )
        x: Float[Tensor, " BS SL NTF"] = (
            shifted_sequence_target_features
            if self.config.teacher_forcing
            else torch.zeros_like(target_sequence_features)
        )
        x: Float[Tensor, " BS SL NIF"] = torch.cat(
            (x, conditioning_sequence_features),
            dim=-1,
        )

        if hasattr(self, "proj_in"):
            x: Float[Tensor, " BS SL HS"] = self.proj_in(x)

        if isinstance(self.model, FNN):
            x: Float[Tensor, " BS SL NOL"] = self.model(x)
        elif isinstance(self.model, Mamba | Mamba2):
            x: Float[Tensor, " BS SL HS"] = self.model(x)
        else:  # isinstance(self.model, RNN | LSTM):
            x: Float[Tensor, " BS SL HS"] = self.model(x)[0]

        if hasattr(self, "proj_out"):
            x: Float[Tensor, " BS SL NOL"] = self.proj_out(x)

        return x

    def predict_sequence_logits_and_features_inference(
        self: "BaseCAM",
        conditioning_sequence_features: Float[Tensor, " BS SL NCF"],
    ) -> tuple[Float[Tensor, " BS SL NOL"], Float[Tensor, " BS SL NTF"]]:
        """Passes `conditioning_sequence_features` through the
        model one timestep at a time in order to predict the sequence
        of logits and features.
        """
        BS, SL, _ = conditioning_sequence_features.shape
        # No previous predicted features to start with, so we use zeroes
        # as placeholders for the first iteration of the
        # autoregressive process.
        previous_predicted_timestep_features = torch.zeros(
            BS,
            1,
            self.config.num_target_features,
        ).to(conditioning_sequence_features.device)
        predicted_sequence_features = torch.empty(
            BS,
            SL,
            self.config.num_target_features,
        ).to(conditioning_sequence_features.device)
        predicted_sequence_logits = torch.empty(
            BS,
            SL,
            self.num_output_logits,
        ).to(conditioning_sequence_features.device)
        self.reset(
            batch_size=BS,
            dtype=conditioning_sequence_features.dtype,
            device=conditioning_sequence_features.device,
        )
        for t in range(SL):
            previous_predicted_timestep_logits = predicted_sequence_logits[
                :,
                t : t + 1,
            ] = self.predict_timestep_logits_inference(
                (
                    previous_predicted_timestep_features
                    if self.config.teacher_forcing
                    else torch.zeros_like(
                        previous_predicted_timestep_features,
                    )
                ),
                conditioning_sequence_features[:, t : t + 1],
            )
            previous_predicted_timestep_features = predicted_sequence_features[
                :,
                t : t + 1,
            ] = self.predict_timestep_features_inference(
                previous_predicted_timestep_logits,
            )
        return predicted_sequence_logits, predicted_sequence_features

    @abstractmethod
    def predict_timestep_features_inference(
        self: "BaseCAM",
        predicted_timestep_logits: Float[Tensor, " BS 1 NOL"],
    ) -> Float[Tensor, " BS 1 NTF"]: ...

    def predict_timestep_logits_inference(
        self: "BaseCAM",
        previous_predicted_timestep_features_or_zeroes: Float[
            Tensor,
            " BS 1 NTF",
        ],
        conditioning_timestep_features: Float[Tensor, " BS 1 NCF"],
    ) -> Float[Tensor, " BS 1 NOL"]:
        """`previous_predicted_timestep_features_or_zeroes` and
        `conditioning_timestep_features` are concatenated before being
        passed through the model.
        """
        x: Float[Tensor, " BS 1 NIF"] = torch.cat(
            (
                previous_predicted_timestep_features_or_zeroes,
                conditioning_timestep_features,
            ),
            dim=-1,
        )

        if hasattr(self, "proj_in"):
            x: Float[Tensor, " BS 1 HS"] = self.proj_in(x)

        if isinstance(self.model, FNN):
            x = self.model(x)
        if isinstance(self.model, Mamba | Mamba2):
            x = rearrange(tensor=x, pattern="BS 1 HS -> BS HS")
            x, self.cache = self.model.step(x, self.cache)
            x = rearrange(tensor=x, pattern="BS HS -> BS 1 HS")
        if isinstance(self.model, RNN | LSTM):
            x, self.cache = self.model(x, self.cache)
        if hasattr(self, "proj_out"):
            x: Float[Tensor, " BS 1 NOL"] = self.proj_out(x)

        return x

    def reset(
        self: "BaseCAM",
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.cache: (
            Tensor | tuple[Tensor, Tensor] | list[tuple[Tensor, Tensor]]
        )
        if isinstance(self.model, Mamba):
            self.cache = [
                (
                    torch.zeros(  # H
                        (
                            batch_size,
                            self.model.config.d_model
                            * self.model.config.expand_factor,
                            self.model.config.d_state,
                        ),
                        dtype=dtype,
                        device=device,
                    ),
                    torch.zeros(  # X
                        (
                            batch_size,
                            self.model.config.d_model
                            * self.model.config.expand_factor,
                            self.model.config.d_conv - 1,
                        ),
                        dtype=dtype,
                        device=device,
                    ),
                )
                for _ in range(self.model.config.n_layers)
            ]
        if isinstance(self.model, Mamba2):
            self.cache = [
                layer.get_empty_cache(batch_size)
                for layer in self.model.layers
            ]
        if isinstance(self.model, RNN):
            self.cache = torch.zeros(
                size=(
                    (self.model.bidirectional + 1) * self.model.num_layers,
                    batch_size,
                    self.model.hidden_size,
                ),
                dtype=dtype,
                device=device,
            )
        if isinstance(self.model, LSTM):
            self.cache = (
                torch.zeros(
                    size=(
                        (self.model.bidirectional + 1) * self.model.num_layers,
                        batch_size,
                        self.model.hidden_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros(
                    size=(
                        (self.model.bidirectional + 1) * self.model.num_layers,
                        batch_size,
                        self.model.hidden_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
            )
