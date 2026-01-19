"""Shapes:

TNN: Total number of nodes (in all networks).
"""

import logging

import torch
from jaxtyping import Bool, Float
from torch import Tensor

log = logging.getLogger(__name__)


class WelfordRunningStandardizer:
    def __init__(
        self: "WelfordRunningStandardizer",
        n_mean_m2_x_z: Float[Tensor, "TNNplus1 5"],
    ):
        self.n_mean_m2_x_z: Float[Tensor, "TNNplus1 5"] = n_mean_m2_x_z
        log.debug("a. Initial n_mean_m2_x_z")
        log.debug(n_mean_m2_x_z)

        # Compile the forward pass
        self._forward = torch.compile(self._forward_impl)

    def _forward_impl(
        self: "WelfordRunningStandardizer",
        x_or_z: Float[Tensor, "TNNplus1"],
        n_mean_m2_x_z: Float[Tensor, "TNNplus1 5"],
    ) -> tuple[Float[Tensor, "TNNplus1"], Float[Tensor, "TNNplus1 5"]]:
        """Compiled forward pass - takes state as input and returns updated state."""

        # 1. Get previous state
        prev_n: Float[Tensor, "TNNplus1"]
        prev_mean: Float[Tensor, "TNNplus1"]
        prev_m2: Float[Tensor, "TNNplus1"]
        prev_x: Float[Tensor, "TNNplus1"]
        prev_z: Float[Tensor, "TNNplus1"]
        prev_n, prev_mean, prev_m2, prev_x, prev_z = n_mean_m2_x_z.unbind(dim=1)

        # 2. Define the update mask
        update_mask: Bool[Tensor, "TNNplus1"] = (x_or_z != 0) & (x_or_z != prev_x) & (x_or_z != prev_z)

        # 3. Calculate potential new values
        n_potential: Float[Tensor, "TNNplus1"] = prev_n + 1.0
        delta: Float[Tensor, "TNNplus1"] = x_or_z - prev_mean
        mean_potential: Float[Tensor, "TNNplus1"] = prev_mean + delta / n_potential
        delta_potential: Float[Tensor, "TNNplus1"] = x_or_z - mean_potential
        m2_potential: Float[Tensor, "TNNplus1"] = prev_m2 + delta * delta_potential

        # 4. Conditionally update the stats
        n: Float[Tensor, "TNNplus1"] = torch.where(update_mask, n_potential, prev_n)
        mean: Float[Tensor, "TNNplus1"] = torch.where(update_mask, mean_potential, prev_mean)
        m2: Float[Tensor, "TNNplus1"] = torch.where(update_mask, m2_potential, prev_m2)

        # 5. Calculate z-score
        variance: Float[Tensor, "TNNplus1"] = m2 / torch.clamp(n - 1, min=1)
        std_dev: Float[Tensor, "TNNplus1"] = torch.sqrt(variance)
        is_valid: Bool[Tensor, "TNNplus1"] = n >= 2
        safe_std_dev: Float[Tensor, "TNNplus1"] = torch.clamp(std_dev, min=1e-8)
        raw_z_score: Float[Tensor, "TNNplus1"] = (x_or_z - mean) / safe_std_dev
        z_score_output: Float[Tensor, "TNNplus1"] = torch.where(is_valid, raw_z_score, torch.zeros_like(raw_z_score))

        # 6. Determine final output
        pass_through_mask: Bool[Tensor, "TNNplus1"] = (x_or_z == 0) | (x_or_z == prev_z)
        final_output: Float[Tensor, "TNNplus1"] = torch.where(pass_through_mask, x_or_z, z_score_output)

        # 7. Update state
        new_prev_x: Float[Tensor, "TNNplus1"] = torch.where(update_mask, x_or_z, prev_x)

        # Stack new state
        new_state: Float[Tensor, "TNNplus1 5"] = torch.stack([n, mean, m2, new_prev_x, final_output], dim=1)

        return final_output, new_state

    def __call__(
        self: "WelfordRunningStandardizer",
        x_or_z: Float[Tensor, "TNNplus1"],
    ) -> Float[Tensor, "TNNplus1"]:
        """
        Processes an input tensor 'x_or_z' containing a mix of old z-scores
        and new raw values.

        - If x_or_z[i] == prev_z[i] (old z-score) or x_or_z[i] == 0,
          stats are not updated.
        - If x_or_z[i] != prev_z[i] (new raw value), stats are updated
          using x_or_z[i].

        Returns a tensor where new raw values have been standardized and
        old z-scores remain the same.
        """
        final_output: Float[Tensor, "TNNplus1"]
        new_state: Float[Tensor, "TNNplus1 5"]
        final_output, new_state = self._forward(x_or_z, self.n_mean_m2_x_z)
        self.n_mean_m2_x_z = new_state
        return final_output
