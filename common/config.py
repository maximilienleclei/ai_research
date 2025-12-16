from dataclasses import dataclass
from typing import Annotated as An

from hydra import conf as hc
from hydra import types as ht
from hydra.experimental.callbacks import LogJobReturnCallback
from hydra_zen import make_config

from common.utils.beartype import ge, not_empty, one_of
from common.utils.hydra_zen import generate_config


@dataclass
class BaseSubtaskConfig:
    """Args:

    output_dir: Path to the ``subtask`` output directory. Every
        artifact generated during the ``subtask`` will be stored
        in this directory.
    data_dir: Path to the data directory. This directory is
        shared between ``task`` runs. It can be used to store
        datasets, pre-trained models, etc.
    """

    output_dir: An[str, not_empty()] = "${hydra:runtime.output_dir}"
    data_dir: An[str, not_empty()] = "${oc.env:AI_RESEARCH_PATH}/data/"
    device: An[str, one_of("cpu", "gpu")] = "cpu"
    seed: An[int, ge(0)] = 0


@dataclass
class BaseHydraConfig(
    make_config(
        bases=(hc.HydraConf,),
        callbacks={"log_job_return": generate_config(LogJobReturnCallback)},
        job=hc.JobConf(
            config=hc.JobConf.JobConfig(
                override_dirname=hc.JobConf.JobConfig.OverrideDirname(
                    kv_sep="~",
                    item_sep="#",
                    exclude_keys=[
                        "task",
                        "project",
                        "trainer.max_epochs",
                        "trainer.max_steps",
                        "config.total_num_gens",
                    ],
                ),
            ),
        ),
        mode=ht.RunMode.MULTIRUN,
        sweep=hc.SweepDir(
            dir="${oc.env:AI_RESEARCH_PATH}/data/${project}/${task}/",
            subdir="overrides#${hydra:job.override_dirname}/",
        ),
    ),
): ...
