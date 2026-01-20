"""Hydra-zen configuration utilities.

Provides pre-configured config generators for Hydra store registration:
- generate_config: Full signature, instantiates target directly
- generate_config_partial: Full signature, returns partial[target]
- generate_config_partial_no_full_sig: Partial without full signature

Usage:
    from common.utils.hydra_zen import generate_config

    store(
        generate_config(MyClass, config=generate_config(MyConfig)),
        name="my_component",
        group="component_group",
    )
"""

from dataclasses import is_dataclass
from typing import Any

from hydra_zen import make_custom_builds_fn
from hydra_zen.wrapper import default_to_config
from omegaconf import OmegaConf

# Creates config that instantiates target with all constructor args exposed
generate_config = make_custom_builds_fn(
    populate_full_signature=True,
    hydra_convert="partial",
)

# Creates config that returns partial[target] for deferred instantiation
generate_config_partial = make_custom_builds_fn(
    zen_partial=True,
    populate_full_signature=True,
    hydra_convert="partial",
)

# Creates partial config without exposing all constructor args
generate_config_partial_no_full_sig = make_custom_builds_fn(
    zen_partial=True,
    populate_full_signature=False,
    hydra_convert="partial",
)


def destructure(x: Any) -> Any:
    """See `discussion <https://github.com/mit-ll-responsible-ai/\
        hydra-zen/discussions/621#discussioncomment-7938326>`_.
    """
    # apply the default auto-config logic of `store`
    x = default_to_config(target=x)
    if is_dataclass(obj=x):
        # Recursively converts:
        # dataclass -> omegaconf-dict (backed by dataclass types)
        return OmegaConf.create(
            obj=OmegaConf.to_container(cfg=OmegaConf.create(obj=x))
        )
    return x
