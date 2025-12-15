************
Installation
************

We assume in this section that you are operating in a Linux environment.
On Windows, please use the Windows Subsystem for Linux (WSL).

1. Clone the repository
-----------------------

.. code-block:: bash

    git clone git@github.com:maximilienleclei/ai_repo.git

2. Define the persistent ``AI_REPO_PATH`` variable
--------------------------------------------------

.. code-block:: bash

    # On Linux
    SHELL_CONFIG_FILE="{$HOME}/.bashrc"
    # On MacOS
    SHELL_CONFIG_FILE="$HOME/.zshrc"

    echo -e "\nexport AI_REPO_PATH=${PWD}/ai_repo" >> "${SHELL_CONFIG_FILE}" \
        && source "${SHELL_CONFIG_FILE}"

3. Fetch your Weights & Biases key
----------------------------------

Copy your `Weights & Biases API key <https://wandb.ai/authorize>`_ to
``ai_repo/WANDB_KEY.txt``.

4. Install Docker
-----------------

.. note::

    Skip if you do not have administrator privileges.

Follow the `official installation guide
<https://docs.docker.com/engine/install/>`_. If you have access to a GPU, make
sure to look up the supplementary installation instructions for your GPU from
the official websites.

5. Pull the image
-----------------

.. code-block:: bash

    # Examples

    # Slim CPU-only image
    docker pull docker.io/mleclei/ai_repo:cpu

    # NVIDIA & AMD GPUs respectively
    docker pull docker.io/mleclei/ai_repo:cuda
    docker pull docker.io/mleclei/ai_repo:rocm

    # On a Slurm cluster
    apptainer build -F ai_repo.sif docker://mleclei/ai_repo:cuda
