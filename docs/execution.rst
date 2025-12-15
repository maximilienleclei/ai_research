*********
Execution
*********

Through Docker
--------------

.. code-block:: bash

    # Example: Train MNIST on NVIDIA GPUs
    docker run --privileged --gpus all --rm -e AI_RESEARCH_PATH=${AI_RESEARCH_PATH} \
               -e PYTHONPATH=${PYTHONPATH}:${AI_RESEARCH_PATH} \
               -v ${AI_RESEARCH_PATH}:${AI_RESEARCH_PATH} -v /dev/shm:/dev/shm \
               -w ${AI_RESEARCH_PATH} mleclei/ai_research:cuda \
               python -m projects.classify_mnist task=mlp

Through Apptainer
-----------------

.. code-block:: bash

    # Example: Train MNIST on the Béluga cluster
    module load apptainer && cd ${AI_RESEARCH_PATH} && export PYTHONPATH=${PYTHONPATH}:${AI_RESEARCH_PATH} && \
        export APPTAINERENV_APPEND_PATH=/opt/software/slurm/bin:/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/apptainer/1.1.8/bin && \
        apptainer exec --no-home -B /etc/passwd -B /etc/slurm/ -B /opt/software/slurm -B /usr/lib64/libmunge.so.2 \
                       -B /cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/apptainer/1.1.8/bin/apptainer \
                       -B /var/run/munge/ --env LD_LIBRARY_PATH=/opt/software/slurm/lib64/slurm  -B $AI_RESEARCH_PATH $SCRATCH/ai_research.sif \
                       python -m projects.classify_mnist task=mlp_beluga
