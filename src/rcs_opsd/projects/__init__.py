"""Project build-system detection and adapter interfaces."""

from .base import (
    FoundryAdapter,
    GenericAdapter,
    HardhatAdapter,
    ProjectAdapter,
    ProjectAdapterRegistry,
)

__all__ = [
    "FoundryAdapter",
    "GenericAdapter",
    "HardhatAdapter",
    "ProjectAdapter",
    "ProjectAdapterRegistry",
]

