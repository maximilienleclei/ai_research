"""This module contains the `DiT1D1D` model, which is a modified
version of the original DiT model (`original_dit.DiT`).

The differences between the two models are as follows:

1. Target signal dimensionality

`DiT` is designed for multi-channel ordered 2D signals
(e.g. RGB images). The `DiT1D1D` model, on the other hand,
is designed to work on multi-channel ordered 1D signals (e.g. stereo
audio).

2. Conditioning signal

`DiT` makes use of an embedding table to transform a
class label into an embedding. The `DiT1D1D` model instead
uses a transformer encoder to encode a multi-channel ordered 1D
conditioning signal.

---

Shapes:

BS: Batch size
NV: Number of vectors
VS: Vector(s) size
IC: Number of input channels (unspecified source)
OC: Number of output channels (unspecified source)
SL: Sequence length (unspecified source)
TNC: Number of channels in the target data
MOC: Number of output channels in `DiT1D1D`
TSL: Sequence length of the target data
CNC: Number of channels in the conditioning data
CSL: Sequence length of the conditioning data
NP: Number of patches
PS: Patch size
ES: Embedding size
HS: Hidden size
"""

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from x_transformers.x_transformers import AttentionLayers

from common.dl.litmodule.nnmodule.cond_diffusion.original_dit import (
    DiTBlock,
    TimestepEmbedder,
    get_1d_sincos_pos_embed_from_grid,
)


def modulate_multichannel_1d(
    x: Float[Tensor, " BS NV VS"],
    shift: Float[Tensor, " BS VS"],
    scale: Float[Tensor, " BS VS"],
) -> Float[Tensor, " BS NV VS"]:
    """Shifts and scales the vector(s)."""
    pattern = "BS VS -> BS 1 VS"
    shift, scale = rearrange(shift, pattern), rearrange(scale, pattern)
    return x * (1 + scale) + shift


class PatchEmbedMultiChannelOrdered1D(nn.Module):
    """1D signal --(conv1D)-> patch-wise vector embeddings.

    See https://arxiv.org/abs/2010.11929 for more details on patch-wise
    embeddings.

    Equivalent to `timm.models.vision_transformer.PatchEmbed`
    when the input signal is 1D rather than 2D."""

    def __init__(
        self: "PatchEmbedMultiChannelOrdered1D",
        seq_len: int,
        in_channels: int,
        embd_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embd_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=(patch_size - seq_len % patch_size) % patch_size,
            padding_mode="replicate",
        )
        self.num_patches = seq_len // patch_size + int(
            seq_len % patch_size > 0,
        )

    def forward(
        self: "PatchEmbedMultiChannelOrdered1D",
        x: Float[Tensor, " BS IC SL"],
    ) -> Float[Tensor, " BS NP ES"]:
        x: Float[Tensor, " BS ES NP"] = self.proj(x)
        return rearrange(x, "BS ES NP -> BS NP ES")


class TransformerEncodeMultiChannelOrdered1D(nn.Module):
    """1D signal --(transformer encoder)-> vector embeddings.

    Whereas `original_dit.LabelEmbedder` embeds class labels,
    this class embeds a multi-channel ordered 1D signal.

    Args:
        desired_num_patches: If this number is greater than
            :paramref:`seq_len`, the number of patches will be set to
            :paramref:`seq_len`
        drop_conditioning_prob: The probability of completely blanking
            out the conditioning data (on a per sample basis). For use
            with techniques like classifier-free guidance.
    """

    def __init__(
        self: "TransformerEncodeMultiChannelOrdered1D",
        seq_len: int,
        in_channels: int,
        embd_size: int,
        desired_num_patches: int,
        encoder: AttentionLayers,
        drop_conditioning_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embd_size = embd_size
        self.drop_conditioning_prob = drop_conditioning_prob
        patch_size = seq_len // desired_num_patches or 1
        self.patch_embed = PatchEmbedMultiChannelOrdered1D(
            seq_len=seq_len,
            in_channels=in_channels,
            embd_size=embd_size,
            patch_size=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embd_size),
            requires_grad=False,
        )
        self.encoder = encoder
        self.proj = nn.Linear(
            self.patch_embed.num_patches * embd_size,
            embd_size,
        )

    def forward(
        self: "TransformerEncodeMultiChannelOrdered1D",
        x: Float[Tensor, " BS IC SL"],
    ) -> Float[Tensor, " BS ES"]:
        if self.training:
            x[: int(len(x) * self.drop_conditioning_prob)] = 0
        x: Float[Tensor, " BS NP ES"] = self.patch_embed(x) + self.pos_embed
        x: Float[Tensor, " BS NP ES"] = self.encoder(x)
        x: Float[Tensor, " BS NPxES"] = rearrange(x, "BS NP ES -> BS (NP ES)")
        x: Float[Tensor, " BS ES"] = self.proj(x)
        return x


class FinalLayer1D(nn.Module):
    """Equivalent to `original_dit.FinalLayer` when the output
    signal is 1D rather than 2D."""

    def __init__(
        self: "FinalLayer1D",
        hidden_size: int,
        patch_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * out_channels,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(
        self: "FinalLayer1D",
        x: Float[Tensor, " BS NP HS"],
        c: Float[Tensor, " BS HS"],
    ) -> Float[Tensor, " BS NP PSxOC"]:
        shift, scale = rearrange(
            self.adaLN_modulation(c),
            "BS (split HS) -> split BS HS",
            split=2,
        )
        x: Float[Tensor, " BS NP HS"] = modulate_multichannel_1d(
            self.norm_final(x),
            shift,
            scale,
        )
        x: Float[Tensor, " BS NP PSxOC"] = self.linear(x)
        return x


class DiT1D1D(nn.Module):
    """DiT model for 1D target and conditioning signal.

    Custom Diffusion Transformer (DiT) model for which 1) the target
    signal is a multi-channel ordered 1D signal (e.g. stereo audio)
    instead of a multi-channel ordered 2D signal (e.g. RGB images), and
    2) the conditioning signal is also a multi-channel ordered 1D signal
    (e.g. stereo audio) instead of a class label.

    In the constructor and `initialize_weights` methods, the
    original code is clearly commented out for the sake of comparison.

    Args:
        input_size: The sequence length of the target 1D signal.
        in_channels: The number of channels for the target 1D signal.
        patch_size: The length of each patch.
        hidden_size: Embedding/hidden size used throughout the model
            and its sub-blocks.
        depth: The number of transformer blocks in both the DiT
            and the conditioning encoder.
        num_heads: The number of attention heads.
        mlp_ratio: By how much to scale the hidden size in the
            MLP sub-blocks.
        drop_conditioning_prob: The probability of completely blanking
            out the conditioning data (on a per sample basis), for use
            with classifier-free guidance.
        conditioning_input_size: The sequence length of the conditioning
            data.
        conditioning_num_channels: The number of channels for the
            conditioning data.
        learn_sigma: Whether to learn the standard deviation of the
            output distribution.
    """

    def __init__(
        self: "DiT1D1D",
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        ### OLD ###
        # class_dropout_prob=0.1,
        # num_classes: int = 1000,
        ### NEW ###
        drop_conditioning_prob: float = 0.1,
        conditioning_input_size: int = 311,
        conditioning_in_channels: int = 513,
        ###########
        *,
        learn_sigma: bool = True,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        ### NEW ###
        self.input_size = input_size
        self.hidden_size = hidden_size
        ###########
        ### OLD ###
        """
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        """
        ### NEW ###
        self.x_embedder = PatchEmbedMultiChannelOrdered1D(
            seq_len=input_size,
            in_channels=in_channels,
            embd_size=hidden_size,
            patch_size=patch_size,
        )
        ###########
        self.t_embedder = TimestepEmbedder(hidden_size)
        ### OLD ###
        """
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob
        )
        """
        ### NEW ###
        self.y_embedder = TransformerEncodeMultiChannelOrdered1D(
            seq_len=conditioning_input_size,
            in_channels=conditioning_in_channels,
            embd_size=hidden_size,
            desired_num_patches=self.x_embedder.num_patches,
            encoder=AttentionLayers(
                dim=hidden_size,
                depth=depth,
                heads=num_heads,
                attn_flash=True,
                ff_glu=True,
            ),
            drop_conditioning_prob=drop_conditioning_prob,
        )
        ###########
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ],
        )
        ### OLD ###
        """
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels
        )
        """
        ### NEW ###
        self.final_layer = FinalLayer1D(
            hidden_size=self.hidden_size,
            patch_size=patch_size,
            out_channels=self.out_channels,
        )
        ###########
        self.initialize_weights()

    def initialize_weights(self: "DiT1D1D") -> None:
        # Initialize transformer layers:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        ### OLD ###
        """
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        """
        ### NEW ###
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim=self.hidden_size,
            pos=np.arange(self.x_embedder.num_patches),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim=self.hidden_size,
            pos=np.arange(self.y_embedder.patch_embed.num_patches),
        )
        self.y_embedder.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )
        ###########
        # Initialize patch_embed like nn.Linear (instead of nn.Conv1d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        ### OLD ###
        """
        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        """
        ### NEW ###
        # Initialize patch_embed like nn.Linear (instead of nn.Conv1d):
        w = self.y_embedder.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.y_embedder.patch_embed.proj.bias, 0)
        ###########

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(
        self: "DiT1D1D",
        x: Float[Tensor, " BS NP PSxOC"],
    ) -> Float[Tensor, " BS OC SL"]:
        """Equivalent to `original_dit.DiT.unpatchify` when the
        output signal is 1D rather than 2D."""
        return rearrange(
            x,
            "BS NP (PS OC) -> BS OC (NP PS)",
            OC=self.out_channels,
        )

    def forward(
        self: "DiT1D1D",
        x: Float[Tensor, " BS TNC TSL"],
        t: Int[Tensor, " BS"],
        y: Float[Tensor, " BS CNC CSL"],
    ) -> Float[Tensor, " BS OC SL"]:
        x: Float[Tensor, " BS NP HS"] = self.x_embedder(x) + self.pos_embed
        t: Float[Tensor, " BS HS"] = self.t_embedder(t)
        y: Float[Tensor, " BS HS"] = self.y_embedder(y)
        c: Float[Tensor, " BS HS"] = t + y
        for block in self.blocks:
            x: Float[Tensor, " BS NP HS"] = block(x, c)
        x: Float[Tensor, " BS NP PSxMOC"] = self.final_layer(x, c)
        x: Float[Tensor, " BS MOC TSL"] = self.unpatchify(x)
        return x

    def forward_with_cfg(
        self: "DiT1D1D",
        x: Float[Tensor, " BS TNC TSL"],
        t: Int[Tensor, " BS"],
        y: Float[Tensor, " BS CNC CSL"],
        cfg_scales: Float[Tensor, " BS"],
    ) -> Float[Tensor, " BS MOC TSL"]:
        """Classifier-free guidance version of `forward` given `cfg_scale`"""
        BS: int = len(x)  # noqa: N806
        x: Float[Tensor, " BSx2 TNC TSL"] = x.repeat(2, 1, 1)
        y: Float[Tensor, " BSx2 CNC CSL"] = y.repeat(2, 1, 1)
        y[BS:] = 0
        t: Int[Tensor, " BSx2"] = t.repeat(2)
        model_out: Float[Tensor, " BSx2 MOC TSL"] = self.forward(x, t, y)
        cond_out: Float[Tensor, " BS MOC TSL"] = model_out[:BS]
        uncond_out: Float[Tensor, " BS MOC TSL"] = model_out[BS:]
        out: Float[Tensor, " BS MOC TSL"] = uncond_out + rearrange(
            cfg_scales,
            "BS -> BS 1 1",
        ) * (cond_out - uncond_out)
        return out
