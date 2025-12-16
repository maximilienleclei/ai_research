import os
from pathlib import Path

import wandb


def login_wandb() -> None:
    wandb_key_path = Path(
        str(os.environ.get("AI_RESEARCH_PATH")) + "/WANDB_KEY.txt",
    )
    if wandb_key_path.exists():
        with wandb_key_path.open(mode="r") as f:
            key = f.read().strip()
        wandb.login(key=key)
    else:
        raise FileNotFoundError("W&B key not found.")
