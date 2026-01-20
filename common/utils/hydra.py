"""Hydra utilities and callbacks for job management."""

import os
import shutil
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra.experimental.callback import Callback
from hydra_plugins.hydra_submitit_launcher.config import (
    LocalQueueConf,
    SlurmQueueConf,
)
from hydra_plugins.hydra_submitit_launcher.submitit_launcher import (
    LocalLauncher,
    SlurmLauncher,
)
from omegaconf import DictConfig, OmegaConf

from common.utils.misc import get_path


class MoveLogsCallback(Callback):
    """Callback to move logs from sweep logs/ to output_dir/logs/ after all jobs complete."""

    def on_job_end(
        self: "MoveLogsCallback", config: DictConfig, job_return: Any, **kwargs: Any
    ) -> None:
        """Store mapping of submitit job ID, log dir, and output dir for later moving."""
        if config and "config" in config and "output_dir" in config.config:
            output_dir = Path(config.config.output_dir)
            hydra_cfg = HydraConfig.get()

            # Get submitit_folder
            submitit_folder = hydra_cfg.launcher.get("submitit_folder")

            if submitit_folder:
                log_source_dir = Path(submitit_folder)
                sweep_dir = Path(hydra_cfg.sweep.dir)

                # Get the submitit job ID from the environment or by checking log files
                submitit_job_id = None
                try:
                    import submitit
                    job_env = submitit.JobEnvironment()
                    submitit_job_id = job_env.job_id
                except Exception:
                    # Fallback: find the job ID from log files in the directory
                    for log_file in log_source_dir.glob("*_log.out"):
                        # Extract job ID from filename like "25_0_log.out"
                        submitit_job_id = log_file.stem.split("_")[0]
                        break

                if submitit_job_id:
                    mapping_file = sweep_dir / ".log_mappings"
                    with open(mapping_file, "a") as f:
                        f.write(f"{submitit_job_id}|{log_source_dir}|{output_dir}\n")

    def on_multirun_end(
        self: "MoveLogsCallback", config: DictConfig, **kwargs: Any
    ) -> None:
        """Move logs after all jobs complete and submitit cleanup is done."""
        # Find the mapping file by looking for .log_mappings in potential sweep dirs
        if "hydra" in config and "sweep" in config.hydra and "dir" in config.hydra.sweep:
            sweep_dir = Path(config.hydra.sweep.dir)
            mapping_file = sweep_dir / ".log_mappings"
            multirun_yaml = sweep_dir / "multirun.yaml"

            if mapping_file.exists():
                log_source_dirs_to_clean = set()

                # Read the mappings and move the logs
                with open(mapping_file) as f:
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) != 3:
                            continue
                        submitit_job_id, log_source_dir_str, output_dir_str = parts
                        log_source_dir = Path(log_source_dir_str)
                        output_dir = Path(output_dir_str)

                        if log_source_dir.exists():
                            # Create logs directory with timestamp in output_dir
                            target_logs_dir = output_dir / "logs" / log_source_dir.name
                            target_logs_dir.mkdir(parents=True, exist_ok=True)

                            # Move only the log files for this specific submitit job
                            for log_file in log_source_dir.glob(f"{submitit_job_id}_*"):
                                target_file = target_logs_dir / log_file.name
                                if not target_file.exists():
                                    shutil.move(str(log_file), str(target_file))

                            # Copy multirun.yaml to this job's log directory
                            if multirun_yaml.exists():
                                target_multirun = target_logs_dir / "multirun.yaml"
                                if not target_multirun.exists():
                                    shutil.copy2(multirun_yaml, target_multirun)

                            # Track source log dirs to clean up later
                            log_source_dirs_to_clean.add(log_source_dir)

                # Delete multirun.yaml from sweep dir
                if multirun_yaml.exists():
                    multirun_yaml.unlink()

                # Delete the now-empty timestamp directories from sweep_dir/logs/
                for log_source_dir in log_source_dirs_to_clean:
                    if log_source_dir.exists():
                        # Check if directory is empty or only has empty subdirs
                        try:
                            if not any(log_source_dir.iterdir()):
                                log_source_dir.rmdir()
                        except OSError:
                            # Directory not empty, skip
                            pass

                # Clean up the mapping file
                mapping_file.unlink()


def get_launcher_config() -> LocalQueueConf | SlurmQueueConf:
    """Get the current Hydra launcher configuration.

    Returns:
        LocalQueueConf or SlurmQueueConf depending on launcher type.

    Raises:
        TypeError: If launcher type is not supported.
    """
    launcher_dict_config: DictConfig = HydraConfig.get().launcher
    launcher_container_config = OmegaConf.to_container(
        cfg=launcher_dict_config,
    )
    if not isinstance(launcher_container_config, dict):
        raise TypeError
    launcher_config_dict = dict(launcher_container_config)
    if launcher_dict_config._target_ == get_path(
        LocalLauncher,
    ):
        return LocalQueueConf(**launcher_config_dict)
    if launcher_dict_config._target_ == get_path(
        SlurmLauncher,
    ):
        return SlurmQueueConf(**launcher_config_dict)
    error_msg = f"Unsupported launcher: {launcher_dict_config._target_}"
    raise TypeError(error_msg)
