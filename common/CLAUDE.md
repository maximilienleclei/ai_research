# Common Module Conventions

This document establishes conventions for all submodules under `common/`. Both `dl/` and `ne/` should follow these patterns.

## Module Structure

Each major submodule (`dl/`, `ne/`) should have:

```
common/{module}/
├── CLAUDE.md       # Module-specific guide for AI agents
├── config.py       # TaskConfig and SubtaskConfig dataclasses
├── constants.py    # Named constants (no magic numbers in code)
├── types.py        # Type aliases for readability (optional)
├── store.py        # Top-level Hydra config registration
├── runner.py       # TaskRunner class
└── {components}/   # Subpackages with their own store.py
```

## Store.py Conventions

Every `store.py` file must have a module docstring explaining:
1. What configs are registered
2. The Hydra group hierarchy
3. How to add new configs

### Template

```python
"""Component configuration store registration.

This module registers configs for [description]:
- item1 (group: group/subgroup)
- item2 (group: group/subgroup)

To add new configs:
1. Create the class/config
2. Register with store(..., name="name", group="group")
3. Override in YAML: `override /group: name`
"""

from hydra_zen import ZenStore

def store_configs(store: ZenStore) -> None:
    """Register all [component] configs to the store.

    Args:
        store: The ZenStore instance to register configs to.
    """
    ...
```

## Docstring Style

Use **Google style** for all docstrings (not NumPy style).

### Config Dataclasses

```python
@dataclass
class MyConfig:
    """Short description of the config.

    Attributes:
        field_name: Description of what this field does.
        another_field: Description with default behavior noted.
    """

    field_name: int
    another_field: bool = True
```

### Classes

```python
class MyClass:
    """Short description of the class.

    Longer description if needed, explaining the design
    philosophy or key behaviors.
    """

    def __init__(self, arg1: int, arg2: str) -> None:
        """Initialize the class.

        Args:
            arg1: Description of arg1.
            arg2: Description of arg2.
        """
```

### Functions/Methods

```python
def my_function(param1: int, param2: str = "default") -> bool:
    """Short description of what the function does.

    Longer description if the function is complex.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When this happens.
    """
```

## Constants

Magic numbers should be extracted to a `constants.py` file with descriptive names:

```python
"""Constants used throughout the module.

This module centralizes magic numbers to improve
code readability and maintainability.
"""

# Group related constants with comments
# =============================================================================
# Batch Size Tuning
# =============================================================================

# Minimum throughput improvement required to prefer larger batch
BATCH_SIZE_EFFICIENCY_THRESHOLD = 1.05
```

Then import and use:
```python
from common.dl.constants import BATCH_SIZE_EFFICIENCY_THRESHOLD

if throughput < baseline * BATCH_SIZE_EFFICIENCY_THRESHOLD:
    ...
```

## Type Hints

### Required for All Code

Every function/method argument must have type hints, including `self`:

```python
# Good
def process(self: "MyClass", data: Tensor) -> Tensor:
    ...

# Bad - missing self type hint
def process(self, data: Tensor) -> Tensor:
    ...
```

### Type Aliases

For complex types used repeatedly, create aliases in `types.py`:

```python
"""Type aliases for the module."""

from typing import TypeAlias

DatasetLike: TypeAlias = Dataset | HFDataset | None
```

## CLAUDE.md Structure

Each module's CLAUDE.md should include:

1. **Architecture Overview**: Directory structure and component relationships
2. **Key Design Patterns**: Important architectural decisions
3. **Configuration System**: How Hydra/store works for this module
4. **Common Tasks**: Step-by-step guides for typical modifications
5. **Gotchas**: Non-obvious behaviors and pitfalls
6. **File Reference**: Annotated directory tree
7. **Maintenance Guidelines**: How to keep the module well-maintained

## Error Handling

Never use bare exceptions. Always include descriptive messages:

```python
# Good
if dataset is None:
    raise AttributeError("Dataset is None. Ensure setup() initializes the dataset.")

# Bad
if dataset is None:
    raise AttributeError
```

## Hydra Config Registration

### Pattern for Components with Config

```python
store(
    generate_config(MyClass, config=generate_config(MyConfig)),
    name="my_component",
    group="component_group",
)
```

### Pattern for Simple Components

```python
store(
    generate_config(MyClass),
    name="my_component",
    group="component_group",
)
```

### Pattern for Partials

```python
store(
    generate_config_partial(MyClass, some_arg="default"),
    name="my_partial",
    group="partial_group",
)
```

## Testing Changes

Always test inside the podman container with GPU flags:

```bash
podman run --rm \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    -v "$(pwd):/app" \
    -w /app \
    -e AI_RESEARCH_PATH=/app \
    localhost/maximilienleclei/ai_research:7800xt \
    python -c "from common.{module}.store import store_configs; print('OK')"
```

## Keeping Documentation in Sync

When modifying code:

1. **Update docstrings** if behavior changes
2. **Update CLAUDE.md** if architecture or patterns change
3. **Update constants.py** if adding new magic numbers
4. **Update store.py docstrings** if adding new configs
