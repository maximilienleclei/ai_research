import time

from common.ne.algo.base import BaseAlgo
from common.ne.config import NeuroevolutionSubtaskConfig
from common.ne.eval.base import BaseEval
from common.ne.popu.base import BasePopu


def evolve(
    algo: BaseAlgo,
    eval: BaseEval,
    popu: BasePopu,
    config: NeuroevolutionSubtaskConfig,
) -> float:
    start_time = time.time()
    generation = 0
    while (time.time() - start_time) / 60 < config.num_minutes:
        popu.nets.mutate()
        fitness_scores = eval(popu)
        algo(popu, fitness_scores)
        print(
            f"Gen {generation}: best = {fitness_scores.max():.2f}, mean = {fitness_scores.mean():.2f}"
        )
        generation += 1

    return None
