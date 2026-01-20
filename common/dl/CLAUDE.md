# Deep Learning Module Guide

## Architecture Overview

This module implements a PyTorch Lightning-based deep learning framework with Hydra configuration management.

```
common/dl/
├── config.py          # DeepLearningTaskConfig, DeepLearningSubtaskConfig
├── constants.py       # Magic numbers (batch size thresholds, etc.)
├── types.py           # Type aliases (DatasetLike, DataLoaderPartial)
├── train.py           # Main train() function
├── runner.py          # DeepLearningTaskRunner
├── store.py           # Top-level Hydra config registration
├── datamodule/
│   └── base.py        # BaseDataModule, BaseDataModuleConfig, Datasets
├── litmodule/
│   ├── base.py        # BaseLitModule, BaseLitModuleConfig
│   ├── classification.py
│   ├── store.py       # Optimizer/scheduler registration
│   ├── utils.py       # Visualization helpers
│   ├── cond1d_target1d/   # 1D→1D prediction modules
│   └── nnmodule/
│       ├── feedforward.py  # FNN (MLP)
│       ├── mamba2.py
│       ├── store.py        # nnmodule registration
│       ├── cond_autoreg/   # Autoregressive models
│       └── cond_diffusion/ # Diffusion models
└── utils/
    ├── lightning.py   # Trainer utils, batch/worker tuning
    └── diffusion/     # Gaussian diffusion utilities
```

## Key Design Patterns

### 1. Separation of PyTorch and Lightning

`BaseLitModule` receives a `nn.Module` (called `nnmodule`) as an argument rather than defining layers directly. This separates:
- **nnmodule**: Pure PyTorch architecture (FNN, Mamba, etc.)
- **LitModule**: Training logic, logging, optimization

```python
class MyLitModule(BaseLitModule):
    def step(self, data, stage):
        # self.nnmodule is the PyTorch model
        output = self.nnmodule(data)
        return loss
```

### 2. Partial Functions for Deferred Instantiation

Optimizers, schedulers, and trainers are passed as `functools.partial` objects. This allows Hydra to configure them before instantiation:

```python
# In config: optimizer is partial[AdamW]
# At runtime: self.optimizer = self.optimizer_partial(params=self.parameters())
```

### 3. Automatic Batch Size and Worker Tuning

When `fixed_per_device_batch_size` is None, the framework automatically:
1. Binary searches for max batch size that fits in GPU VRAM
2. Measures GPU processing time per batch
3. Finds minimum workers needed to keep GPU saturated

Key constants in `constants.py`:
- `BATCH_SIZE_EFFICIENCY_THRESHOLD = 1.05`: Larger batch must be 5% faster
- `WORKER_SATURATION_BUFFER = 1.1`: Workers are "good enough" at 110% of GPU speed

## Hydra/Store Configuration System

### Config Hierarchy

```
DeepLearningTaskConfig
├── trainer (group: trainer)
├── datamodule (project-specific)
├── litmodule (project-specific)
│   ├── nnmodule (group: litmodule/nnmodule) → default: fnn
│   ├── optimizer (group: litmodule/optimizer) → default: adamw
│   └── scheduler (group: litmodule/scheduler) → default: constant
├── logger (group: logger)
└── config: DeepLearningSubtaskConfig
```

### Default Configs (from DeepLearningTaskConfig.defaults)

```python
defaults = [
    "_self_",
    {"trainer": "base"},
    {"litmodule/nnmodule": "fnn"},
    {"litmodule/scheduler": "constant"},
    {"litmodule/optimizer": "adamw"},
    {"logger": "wandb"},
    ...
]
```

### How to Override in YAML

```yaml
defaults:
  - override /litmodule/nnmodule: mamba2
  - override /litmodule/optimizer: adam
```

## Common Tasks

### Adding a New Neural Network Architecture

1. Create the module in `litmodule/nnmodule/`:
```python
@dataclass
class MyNetConfig:
    hidden_size: int = 256

class MyNet(nn.Module):
    def __init__(self, config: MyNetConfig):
        ...
```

2. Register in `litmodule/nnmodule/store.py`:
```python
store(
    generate_config(MyNet, config=generate_config(MyNetConfig)),
    name="mynet",
    group="litmodule/nnmodule",
)
```

3. Use in YAML: `override /litmodule/nnmodule: mynet`

### Adding a New LitModule

1. Subclass `BaseLitModule` and implement `step()`:
```python
class MyLitModule(BaseLitModule):
    def step(self, data, stage) -> Tensor:
        x, y = data
        pred = self.nnmodule(x)
        return F.mse_loss(pred, y)
```

2. Register in project's store (NOT in common/dl):
```python
store(
    generate_config(MyLitModule, config=generate_config(MyLitModuleConfig)),
    name="my_litmodule",
    group="litmodule",
)
```

**Important**: Do NOT include nnmodule/optimizer/scheduler in the registration - those come from defaults.

### Adding a New DataModule

1. Subclass `BaseDataModule`:
```python
class MyDataModule(BaseDataModule):
    def prepare_data(self):
        # Download data

    def setup(self, stage):
        self.datasets.train = MyDataset(...)
        self.datasets.val = MyDataset(...)
```

2. Register in project's store with the dataloader partial.

## Important Implementation Details

### W&B Logging

`BaseLitModule` maintains two tables (`wandb_train_table`, `wandb_val_table`) for logging samples. Override `update_wandb_data_before_log()` to transform data before logging:

```python
def update_wandb_data_before_log(self, data, stage):
    for d in data:
        d["image"] = wandb.Image(d["x"])  # Convert tensor to image
```

### Checkpoint State

The datamodule saves `per_device_batch_size` and `per_device_num_workers` to checkpoints, so tuned values persist across restarts.

### torch.compile Support

Enabled when:
- `config.compile = True`
- `config.device = "gpu"`
- CUDA compute capability >= 7 (defined in `constants.py`)

## Conditional Autoregressive Models (BaseCAM)

The `cond_autoreg/base.py` module is complex. Key points:

1. **Supports multiple backends**: FNN, RNN, LSTM, Mamba, Mamba2
2. **Teacher forcing**: During training, uses ground truth from previous timestep
3. **Inference caching**: Stateful models (RNN/LSTM/Mamba) maintain cache for efficient sequential generation
4. **Projection layers**: `proj_in`/`proj_out` added automatically based on model type

Cache structure varies by model type - see `reset()` method docstring.

## Maintenance Guidelines

### When Adding New Code

1. **Type hints**: Every function argument must have type hints, including `self`
2. **Docstrings**: Add docstrings to all public classes, methods, and configs following the Attributes pattern:
   ```python
   @dataclass
   class MyConfig:
       """Description.

       Attributes:
           field_name: What this field does.
       """
   ```
3. **Constants**: Extract magic numbers to `constants.py` with descriptive names
4. **Store registration**: Update the relevant `store.py` and add to module docstring

### When Modifying Existing Code

1. Update docstrings if behavior changes
2. If adding new constants, add them to `constants.py`
3. Keep this CLAUDE.md up-to-date with architectural changes

### Code Quality Checks

Run inside podman container:
```bash
podman run --rm \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --ipc=host \
    -v "$(pwd):/app" -w /app -e AI_RESEARCH_PATH=/app \
    localhost/maximilienleclei/ai_research:7800xt \
    python -c "from common.dl.store import store_configs; print('OK')"
```

### File Organization

- `constants.py`: Numeric constants with semantic names
- `types.py`: Type aliases for improved readability
- `store.py` files: Only config registration, no business logic
- Base classes: In `base.py` files within each subpackage
