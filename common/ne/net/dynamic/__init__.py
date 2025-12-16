"""Dynamic topology neural networks for neuroevolution.

This package contains networks with evolving topology that can grow and prune nodes
and connections through evolution. Uses graph-based recurrence and online standardization.
"""

from .evolution import Net, Node, NodeList
from .population import DynamicNetPopulation

__all__ = [
    "Net",
    "Node",
    "NodeList",
    "DynamicNetPopulation",
]
