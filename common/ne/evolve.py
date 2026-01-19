import logging
import time

from common.ne.algo.base import BaseAlgo
from common.ne.config import NeuroevolutionSubtaskConfig
from common.ne.eval.base import BaseEval
from common.ne.popu.base import BasePopu
from common.utils.misc import seed_all

log = logging.getLogger(__name__)


def evolve(
    algo: BaseAlgo,
    eval: BaseEval,
    popu: BasePopu,
    config: NeuroevolutionSubtaskConfig,
) -> float:
    seed_all(config.seed)
    log.info(config.output_dir[config.output_dir.find("results/") :])
    start_time = time.time()
    generation = 0
    while (time.time() - start_time) / 60 < config.num_minutes:
        popu.nets.mutate()
        fitness_scores = eval(popu, generation)
        algo(popu, fitness_scores)
        # Log fitness and env rewards if available
        msg = f"Gen {generation}: fitness best={fitness_scores.max():.2f} mean={fitness_scores.mean():.2f}"
        if hasattr(eval, "last_fitness_G"):
            fitness_G = eval.last_fitness_G
            msg += f" | fitness_G best={fitness_G.max():.2f} mean={fitness_G.mean():.2f}"
        if hasattr(eval, "last_fitness_D"):
            fitness_D = eval.last_fitness_D
            msg += f" | fitness_D best={fitness_D.max():.2f} mean={fitness_D.mean():.2f}"
        if hasattr(eval, "last_env_rewards"):
            env_rewards = eval.last_env_rewards
            msg += f" | env_reward best={env_rewards.max():.1f} mean={env_rewards.mean():.1f}"
        if hasattr(eval, "last_target_env_rewards"):
            target_rewards = eval.last_target_env_rewards
            msg += f" | target_reward={target_rewards.mean():.1f}"
        print(msg)
        generation += 1

    return None
