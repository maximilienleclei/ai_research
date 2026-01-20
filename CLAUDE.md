# Development Guidelines

## Code Quality Requirements
- Type hints every function/method argument, including `self`
- When running projects/tasks, the execution results (logs, checkpoints, etc) are in the project folder, in result/
- Log files are at: `projects/{project}/results/{task}/logs/{timestamp}/*_log.out`
  To read the latest log (auto-samples if >50 lines):
  ```bash
  F=$(ls -t projects/{project}/results/{task}/logs/*/*.out 2>/dev/null | head -1) && L=$(wc -l < "$F") && I=$((L / 20)) && [ $I -lt 1 ] && I=1 && [ $L -le 50 ] && cat "$F" || awk "NR<=5 || NR%$I==0 || NR==$L" "$F"
  ```

## Running Projects
Example podman command:
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
    python -m projects.{project} task={task}
```

For imitation learning projects (act_pred, adv_gen) that need stable-baselines3:
```bash
podman run --rm \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    -v "$(pwd):/app" \
    -v "$HOME/rl-baselines3-zoo:/rl-baselines3-zoo:ro" \
    -w /app \
    -e AI_RESEARCH_PATH=/app \
    -e RL_ZOO_PATH=/rl-baselines3-zoo \
    localhost/maximilienleclei/ai_research:7800xt \
    python -m projects.{project} task={task}
```

**NOTE:** DO NOT CONSIDER BACKWARD COMPATIBILITY, ALWAYS KEEP THE CODEBASE CLEAN IS THE PRIORITY.

**NOTE 2:** Always make sure `device` is always set to the GPU.

---

# Codebase Tree

This AI research framework looks at Deep Learning and Neuroevolution approaches for control and prediction tasks.

## Directory Structure Overview

```
ai_research/
├── common/
│ ├── dl/ ~~~~~ Deep Learning framework
│ │ ├── datamodule/
│ │ │ └── base.py
│ │ ├── litmodule/
│ │ │ ├── cond1d_target1d/ ~~~~~ 1D input vectors → 1D output vectors
│ │ │ │ ├── autoregression.py
│ │ │ │ ├── base.py
│ │ │ │ ├── diffusion.py
│ │ │ │ └── store.py
│ │ │ ├── nnmodule/
│ │ │ │ ├── cond_autoreg/ ~~~~~ Conditional autoregression
│ │ │ │ │ ├── base.py
│ │ │ │ │ ├── discriminative.py
│ │ │ │ │ ├── generative.py
│ │ │ │ │ └── store.py
│ │ │ │ ├── cond_diffusion/ ~~~~~ Conditional Diffusion
│ │ │ │ │ ├── dit_1d_1d.py ~~~~~ 1D input vectors → 1D output vectors
│ │ │ │ │ ├── original_dit.py ~~~~~ Diffusion transformer
│ │ │ │ │ └── store.py
│ │ │ │ ├── feedforward.py
│ │ │ │ ├── mamba2.py
│ │ │ │ └── store.py
│ │ │ ├── base.py
│ │ │ ├── classification.py
│ │ │ ├── store.py
│ │ │ └── utils.py
│ │ ├── utils/
│ │ │ ├── diffusion/
│ │ │ └── lightning.py
│ │ ├── config.py ~~~~~ DeepLearningTaskConfig & DeepLearningSubtaskConfig 
│ │ ├── runner.py ~~~~~ DeepLearningTaskRunner
│ │ ├── store.py
│ │ └── train.py ~~~~~ train()
│ ├── ne/ ~~~~~ Neuroevolution framework (see common/ne/CLAUDE.md for details)
│ │ ├── algo/
│ │ │ ├── base.py
│ │ │ ├── simple_ga.py ~~~~~ 50% truncation selection, no crossovers
│ │ │ └── store.py
│ │ ├── eval/
│ │ │ ├── score/ ~~~~~ Environment reward-based fitness
│ │ │ │ ├── base.py
│ │ │ │ ├── gym.py
│ │ │ │ └── torchrl.py
│ │ │ ├── act_pred.py ~~~~~ Action prediction (behavior cloning from SB3 agents)
│ │ │ ├── adv_gen.py ~~~~~ Adversarial generation (imitation learning)
│ │ │ ├── human_act_pred.py ~~~~~ Action prediction from human behavior data (supports CL info)
│ │ │ ├── base.py
│ │ │ └── store.py
│ │ ├── popu/
│ │ │ ├── nets/
│ │ │ │ ├── static/
│ │ │ │ │ ├── base.py
│ │ │ │ │ ├── feedforward.py
│ │ │ │ │ └── recurrent.py
│ │ │ │ ├── dynamic/ ~~~~~ Networks whose architecture evolves over time
│ │ │ │ │ ├── base.py ~~~~~ Multi-network wrapper around evolution.py
│ │ │ │ │ ├── evolution.py ~~~~~ Single network mutation rules
│ │ │ │ │ └── utils.py
│ │ │ │ ├── base.py
│ │ │ │ └── store.py
│ │ │ ├── actor.py ~~~~~ Action-taking population
│ │ │ ├── adv_gen.py ~~~~~ Double-function population: action-taking and discrimination
│ │ │ ├── base.py
│ │ │ └── store.py
│ │ ├── config.py ~~~~~ NeuroevolutionTaskConfig & NeuroevolutionSubtaskConfig
│ │ ├── evolve.py ~~~~~ evolve()
│ │ ├── runner.py ~~~~~ NeuroevolutionTaskRunner
│ │ └── store.py
│ ├── utils/
│ │ ├── beartype.py ~~~~~ Type hints: not_empty, equal, one_of, ge, gt, le, lt
│ │ ├── hydra_zen.py ~~~~~ store.py utils: generate_config, generate_config_partial, generate_config_partial_no_full_sig
│ │ ├── hydra.py
│ │ ├── misc.py
│ │ ├── runner.py
│ │ ├── torch.py
│ │ └── wandb.py
│ ├── config.py ~~~~~ BaseSubtaskConfig, BaseHydraConfig
│ ├── runner.py ~~~~~ BaseTaskRunner
│ └── store.py
├── data/
│ └── human_behaviour_control_tasks/
│   ├── collect.py
│   ├── data_analysis.py
│   ├── replay.py
│   ├── training_sessions_plot.png
│   ├── sub01_data_acrobot.json
│   ├── sub01_data_cartpole.json
│   ├── sub01_data_lunarlander.json
│   ├── sub01_data_mountaincar.json
│   ├── sub02_data_acrobot.json
│   ├── sub02_data_cartpole.json
│   ├── sub02_data_lunarlander.json
│   └── sub02_data_mountaincar.json
├── projects/
│ ├── dl_classify_mnist/ ~~~~~ Toy DL example
│ ├── ne_control_act_pred/ ~~~~~ NE with action prediction (behavior cloning from SB3)
│ ├── ne_control_adv_gen/ ~~~~~ NE with adversarial generation
│ ├── ne_control_score_encoded/ ~~~~~ NE with autoencoder-encoded states
│ ├── ne_control_human_act_pred/ ~~~~~ NE with human behavior cloning (supports CL info)
│ ├── ne_control_score/ ~~~~~ Neuroevolution for control
│ └── ift6167/                        # Academic research project
├── README.md
├── CLAUDE.md
├── requirements.txt
├── Dockerfile
├── .devcontainer.json
└── PROMPT.txt
```

* Make sure to look at the existing projects to see how they are set up.
* Make sure you understand how store.py files work.

## Hydra/Store Configuration Patterns

**Key learnings when creating new DL projects:**

1. **Default configs are defined in `common/dl/config.py`** (`DeepLearningTaskConfig.defaults`):
   - `litmodule/nnmodule: fnn` - default nnmodule
   - `litmodule/scheduler: constant` - default scheduler
   - `litmodule/optimizer: adamw` - default optimizer

2. **To use a custom nnmodule**, you must:
   - Register it in the store under `group="litmodule/nnmodule"`
   - Override the default in your YAML: `override /litmodule/nnmodule: your_nnmodule`

3. **Litmodule registration should NOT include nnmodule/optimizer/scheduler** - those come from defaults. Compare:
   ```python
   # WRONG - verbose and causes config merging issues
   store(generate_config(MyLitModule, nnmodule=..., optimizer=...), ...)

   # CORRECT - let defaults handle nnmodule/optimizer/scheduler
   store(generate_config(MyLitModule, config=generate_config(MyConfig)), ...)
   ```

4. **Dataclass configs with required fields**: If parent class has defaults, child fields must also have defaults to avoid dataclass ordering errors. Use `"???"` for Hydra-required fields:
   ```python
   @dataclass
   class MyConfig(BaseConfig):  # BaseConfig has defaults
       my_field: str = "???"   # Must have default, use "???" for required
   ```

5. **Hydra output directory structure**: The output subdir is `${hydra:job.override_dirname}` which is built from CLI overrides. Keys in `exclude_keys` (see `common/config.py`) don't create subdirs:
   - `trainer.max_epochs` is EXCLUDED → no subdir, checkpoints go to timestamp folder
   - `trainer.max_time` is NOT excluded → creates subdir like `trainer.max_time~{minutes:1}/`
   - Use `trainer.max_time` instead of `trainer.max_epochs` to get proper checkpoint paths


## Execution Flow

1. Entry point: `projects/{project}/__main__.py` or `train.py`
2. Instantiates TaskRunner (DL or NE)
3. Hydra loads YAML configs from `task/` directory
4. Calls `TaskRunner.run_task()` which delegates to `run_subtask()`
5. Results saved to `projects/{project}/results/{task}/`
6. WandB logs metrics and artifacts (if enabled)

**IMPORTANT:** Keep this file up-to-date as the codebase evolves.
