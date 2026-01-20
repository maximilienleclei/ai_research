from dataclasses import dataclass


@dataclass
class ScoreEvalConfig:
    env_name: str
    max_steps: int = 500
    num_workers: int = "${popu.config.size}"
    seed: int = "${config.seed}"
    device: str = "${config.device}"
