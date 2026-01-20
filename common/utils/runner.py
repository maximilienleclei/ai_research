"""Utilities for extracting project and task info from CLI arguments."""

import sys
from pathlib import Path


def get_task_name() -> str:
    """Extract task name from CLI arguments (task=<name>)."""
    has_task_arg = False
    for arg in sys.argv:
        if arg.startswith("task="):
            has_task_arg = True
            task_name = arg.split("=", maxsplit=1)[-1]
    if not has_task_arg:
        error_msg = (
            "You must specify the ``task`` argument in the form"
            "``task=my_task``."
        )
        raise RuntimeError(error_msg)
    return task_name


def get_project_name() -> str:
    """Get project name from the main file's parent directory."""
    main_file = str(sys.modules["__main__"].__file__)
    main_file_path = Path(main_file).resolve()
    return str(main_file_path.parent.name)


def get_absolute_project_path() -> str:
    """Get absolute path to project directory.

    Raises:
        RuntimeError: If main file is not in a directory with a task/ subdirectory.
    """
    main_file = str(sys.modules["__main__"].__file__)
    main_file_path = Path(main_file).resolve()
    main_file_parent_path = main_file_path.parent
    if not (main_file_parent_path / "task").exists():
        error_msg = (
            "The main file must be located in the same directory as the "
            "``task`` directory."
        )
        raise RuntimeError(error_msg)
    return str(main_file_parent_path)
