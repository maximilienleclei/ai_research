"""Shapes:

BS: Batch size
SL: Sequence length
SLM1: Sequence length minus 1
NTF: Number of target features
NCF: Number of conditioning features
NIF: Number of input features
NOL: Number of output logits
NG: Number of gaussians
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as f
from einops import rearrange, reduce, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.distributions.normal import Normal

from common.dl.litmodule.nnmodule.cond_autoreg.base import (
    BaseCAM,
    BaseCAMConfig,
)


@dataclass
class CGAMConfig(BaseCAMConfig):
    num_gaussians: int = 3


class CGAM(BaseCAM):
    """A conditional Gaussian Mixture Model (GMM) inspired by
    "Generating Sequences With Recurrent Neural Networks" (Graves, 2013)
    https://arxiv.org/abs/1308.0850.
    """

    def __init__(
        self: "CGAM",
        config: CGAMConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.num_output_logits = config.num_gaussians * (
            1 + config.num_target_features * 2
        )
        super().__init__(config, *args, **kwargs)
        self.config: CGAMConfig
        self.norm = nn.RMSNorm(self.num_output_logits)

    def compute_loss(
        self: "CGAM",
        predicted_logits: Float[Tensor, " BS SL NOL"],
        target_features: Float[Tensor, " BS SL NTF"],
    ) -> Float[Tensor, " "]:
        log_pi, mu, sigma = self.convert_logits_to_gaussian_mixture_parameters(
            predicted_logits,
        )
        BS, SL, _ = log_pi.shape
        target_features = repeat(
            tensor=target_features,
            pattern="BS SL NTF -> BS SL 1 NTF",
        )
        gaussian_distribution = Normal(loc=mu, scale=sigma)
        log_prob: Float[Tensor, " BS SL NG NTF"] = (
            gaussian_distribution.log_prob(value=target_features)
        )
        log_prob_per_gaussian = reduce(
            tensor=log_prob,
            pattern="BS SL NG NTF -> BS SL NG",
            reduction=torch.sum,
        )
        weighted_log_prob_per_gaussian: Float[Tensor, " BS SL NG"] = (
            log_pi + log_prob_per_gaussian
        )
        log_likelihood: Float[Tensor, " BS SL"] = reduce(
            tensor=weighted_log_prob_per_gaussian,
            pattern="BS SL NG -> BS SL",
            reduction=torch.logsumexp,
        )
        return -log_likelihood.mean()

    def predict_sequence_logits_and_features_fitting(
        self: "CGAM",
        conditioning_sequence_features: Float[Tensor, " BS SL NCF"],
        target_sequence_features: Float[Tensor, " BS SL NTF"],
    ) -> tuple[Float[Tensor, " BS SL NOL"], Float[Tensor, " BS SL NTF"]]:
        predicted_sequence_logits = self.predict_sequence_logits_fitting(
            conditioning_sequence_features,
            target_sequence_features,
        )
        predicted_sequence_features = self.predict_features(
            predicted_sequence_logits,
        )
        return predicted_sequence_logits, predicted_sequence_features

    def predict_timestep_features_inference(
        self: "CGAM",
        predicted_timestep_logits: Float[Tensor, " BS 1 NOL"],
    ) -> Float[Tensor, " BS 1 NTF"]:
        return self.predict_features(predicted_timestep_logits)

    def predict_features(
        self: "CGAM",
        predicted_logits: (
            Float[Tensor, " BS SL NOL"] | Float[Tensor, " BS 1 NOL"]
        ),
    ) -> Float[Tensor, " BS SL NTF"] | Float[Tensor, " BS 1 NTF"]:
        log_pi, mu, sigma = self.convert_logits_to_gaussian_mixture_parameters(
            predicted_logits,
        )
        BS, SL, _, NTF = mu.shape
        pi: Float[Tensor, " BS SL NG"] = torch.exp(log_pi)
        pi = pi.clamp_min(1e-8)
        pi = pi / (pi.sum(dim=-1, keepdim=True) + 1e-8)
        flat_pi = rearrange(
            tensor=pi,
            pattern="BS SL NG -> (BS SL) NG",
        )  # Required for the `torch.multinomial` operation
        flat_indices: Int[Tensor, " BSxSL 1"] = torch.multinomial(
            input=flat_pi,
            num_samples=1,
        )
        indices = rearrange(
            tensor=flat_indices,
            pattern="(BS SL) 1 -> BS SL 1",
            BS=BS,
            SL=SL,
        )
        repeated_indices = repeat(
            tensor=indices,
            pattern="BS SL 1 -> BS SL 1 NTF",
            NTF=NTF,
        )
        sampled_mu: Float[Tensor, " BS SL NTF"] = torch.gather(
            input=mu,
            dim=2,
            index=repeated_indices,
        ).squeeze(dim=-2)
        sampled_sigma: Float[Tensor, " BS SL NTF"] = torch.gather(
            input=sigma,  # BS SL NG NTF
            dim=2,  # NG
            index=repeated_indices,
        ).squeeze(dim=-2)
        return sampled_mu + sampled_sigma * torch.randn_like(sampled_sigma)

    def convert_logits_to_gaussian_mixture_parameters(
        self: "CGAM",
        predicted_logits: Float[Tensor, " BS SL NOL"],
    ) -> tuple[
        Float[Tensor, " BS SL NG"],
        Float[Tensor, " BS SL NG NTF"],
        Float[Tensor, " BS SL NG NTF"],
    ]:
        x = self.norm(predicted_logits)
        pi_logits: Float[Tensor, " BS SL NG"] = x[
            ...,
            : self.config.num_gaussians,
        ]
        mu: Float[Tensor, " BS SL NGxNTF"] = x[
            ...,
            self.config.num_gaussians : self.config.num_gaussians
            + self.config.num_gaussians * self.config.num_target_features,
        ]
        log_sigma: Float[Tensor, " BS SL NGxNTF"] = x[
            ...,
            self.config.num_gaussians
            + self.config.num_gaussians * self.config.num_target_features :,
        ]
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=10.0)
        # temperature goes here, replace with pi_logits / temperature
        log_pi: Float[Tensor, " BS SL NG"] = f.log_softmax(pi_logits, dim=-1)
        mu: Float[Tensor, " BS SL NG NTF"] = rearrange(
            tensor=mu,
            pattern="BS SL (NG NTF) -> BS SL NG NTF",
            NG=self.config.num_gaussians,
        )
        log_sigma: Float[Tensor, " BS SL NG NTF"] = rearrange(
            tensor=log_sigma,
            pattern="BS SL (NG NTF) -> BS SL NG NTF",
            NG=self.config.num_gaussians,
        )
        sigma: Float[Tensor, " BS SL NG NTF"] = torch.exp(log_sigma)
        return log_pi, mu, sigma
