# Loosely based on https://github.com/DLR-RM/stable-baselines3/blob/master/Dockerfile

# Use the micromamba image with version 2.0.4 based on Ubuntu 22.04
FROM mambaorg/micromamba:2.0-ubuntu22.04
# With cuda:
#FROM mambaorg/micromamba:2.0-cuda12.5.1-ubuntu22.04

# Activate micromamba environment automatically
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Set mujoco headless
USER root

ENV MUJOCO_GL="egl"
ENV PYOPENGL_PLATFORM="egl"

RUN apt-get update && \
    apt-get install -y \
        libegl1 \
        libegl-mesa0 \
        libgles2-mesa \
        libgl1-mesa-dri && \
    rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER


# Set the working directory
WORKDIR /quad-rl

# Copy your project files into the container
COPY . .

# Install Python 3.10, pip, and PyTorch CPU-only version using micromamba
RUN micromamba install -n base -y python=3.10 \
    pytorch cpuonly -c conda-forge -c pytorch -c nvidia && \
    micromamba clean --all --yes

# Install the package in editable mode
RUN pip install -e . && \
    # Use headless version for docker
    pip uninstall -y opencv-python && \
    pip install opencv-python-headless && \
    pip cache purge

# Set the entrypoint to provide an interactive shell
ENTRYPOINT ["/bin/bash"]

# Default command
CMD ["-i"]