"""Shapes:

TNN: Total number of nodes (in all networks).
"""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


class WelfordRunningStandardizer:
    def __init__(
        self: "WelfordRunningStandardizer",
        n_mean_m2_x_z: Float[Tensor, "TNNplus1 5"],
        verbose: bool = False,
    ):
        self.n_mean_m2_x_z: Float[Tensor, "TNNplus1 5"] = n_mean_m2_x_z
        self.verbose = verbose
        if verbose:
            print("a. Initial n_mean_m2_x_z")
            print(n_mean_m2_x_z)

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
        # 1. Get previous state
        prev_n: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x_z[:, 0]
        prev_mean: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x_z[:, 1]
        prev_m2: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x_z[:, 2]
        prev_x: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x_z[:, 3]
        prev_z: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x_z[:, 4]

        # 2. Define the update mask. Update only for new values.
        update_mask: Bool[Tensor, "TNNplus1"] = (
            (x_or_z != 0) & (x_or_z != prev_x) & (x_or_z != prev_z)
        )

        if self.verbose:
            print("b. x_or_z (input)")
            print(x_or_z)
            print("c. prev_x (previous raw output)")
            print(prev_x)
            print("c. prev_z (previous z-score output)")
            print(prev_z)
            print("d. update_mask (update=True)")
            print(update_mask)

        # 3. Calculate potential new values for the stats from `x_or_z`.
        n_potential = prev_n + 1.0
        delta = x_or_z - prev_mean
        mean_potential = prev_mean + delta / n_potential
        delta_potential = x_or_z - mean_potential
        m2_potential = prev_m2 + delta * delta_potential

        # 4. Conditionally update the stats.
        n = self.n_mean_m2_x_z[:, 0] = torch.where(
            update_mask, n_potential, prev_n
        )
        mean = self.n_mean_m2_x_z[:, 1] = torch.where(
            update_mask, mean_potential, prev_mean
        )
        m2 = self.n_mean_m2_x_z[:, 2] = torch.where(
            update_mask, m2_potential, prev_m2
        )

        # 5. Calculate z-score using the updated stats.
        variance: Float[Tensor, "TNNplus1"] = m2 / n
        std_dev: Float[Tensor, "TNNplus1"] = torch.sqrt(variance)
        is_valid: Bool[Tensor, "TNNplus1"] = n >= 2
        safe_std_dev: Float[Tensor, "TNNplus1"] = torch.clamp(
            std_dev, min=1e-8
        )
        raw_z_score: Float[Tensor, "TNNplus1"] = (x_or_z - mean) / safe_std_dev
        z_score_output: Float[Tensor, "TNNplus1"] = torch.where(
            is_valid, raw_z_score, torch.tensor(0.0)
        )
        if self.verbose:
            print("e. z_score_output")
            print(z_score_output)

        pass_through_mask: Bool[Tensor, "TNNplus1"] = (x_or_z == 0) | (
            x_or_z == prev_z
        )
        if self.verbose:
            print("f. pass_through_mask")
            print(pass_through_mask)

        # 6. Determine the final output.
        final_output: Float[Tensor, "TNNplus1"] = torch.where(
            pass_through_mask, x_or_z, z_score_output
        )

        # 7. Store the state for the next call.
        self.n_mean_m2_x_z[:, 3] = torch.where(update_mask, x_or_z, prev_x)
        self.n_mean_m2_x_z[:, 4] = final_output

        if self.verbose:
            print("g. Final n_mean_m2_x_z (state)")
            print(self.n_mean_m2_x_z)

        # 8. Return the final, processed tensor
        return final_output.clone()
