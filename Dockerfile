# syntax=docker/dockerfile:experimental
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# nvidia/cuda:10.1-base-ubuntu18.04
# ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python3-setuptools \
    python3-pip \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    pkg-config

WORKDIR /src

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH /root/miniconda3/bin:$PATH

ENV CONDA_PREFIX /root/miniconda3/envs/torchbeast

# Clear .bashrc (it refuses to run non-interactively otherwise).
RUN echo > ~/.bashrc

# Add conda logic to .bashrc.
RUN conda init bash

# Create new environment and install some dependencies.
RUN conda create -y -n torchbeast python=3.7 \
    protobuf \
    numpy \
    ninja \
    pyyaml \
    mkl \
    mkl-include \
    setuptools \
    cmake \
    cffi \
    typing

# Activate environment in .bashrc.
RUN echo "conda activate torchbeast" >> /root/.bashrc

# Make bash excecute .bashrc even when running non-interactively.
ENV BASH_ENV /root/.bashrc

# Install PyTorch.

# Would like to install PyTorch via pip. Unfortunately, there's binary
# incompatability issues (https://github.com/pytorch/pytorch/issues/18128).
# Otherwise, this would work:
# # # Install PyTorch. This needs increased Docker memory.
# # # (https://github.com/pytorch/pytorch/issues/1022)
# # RUN pip download torch
# # RUN pip install torch*.whl

# Added (referencing https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile)
RUN conda install -y -c pytorch magma-cuda101

RUN git clone --single-branch --branch v1.2.0 --recursive https://github.com/pytorch/pytorch

WORKDIR /src/pytorch

ENV CMAKE_PREFIX_PATH ${CONDA_PREFIX}

# Added (referencing https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile)
RUN git submodule update --init --recursive

# Added (referencing https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile)
RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
#     pip install -v .

RUN python setup.py install

# Clone TorchBeast.
WORKDIR /src/torchbeast

COPY .git /src/torchbeast/.git

RUN git reset --hard

# Collect and install grpc.
RUN git submodule update --init --recursive

RUN ./scripts/install_grpc.sh

# Install nest.
RUN pip install nest/

# Install PolyBeast's requirements.
RUN pip install -r requirements.txt

# Compile libtorchbeast.
ENV LD_LIBRARY_PATH ${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

RUN python setup.py install

ENV OMP_NUM_THREADS 1

# CMD ["nvidia-smi"]

# Small Run.
CMD ["bash", "-c", "python -m torchbeast.polybeast \
       --num_actors 10 \
       --total_steps 2_000 \
       --learning_rate 0.0002 \
       --grad_norm_clipping 1280 \
       --epsilon 0.01 \
       --entropy_cost 0.01 \
       --unroll_length 50 --batch_size 32"]

# # Final Run.
# CMD ["bash", "-c", "python -m torchbeast.polybeast \
#        --num_actors 10 \
#        --total_steps 2_000_000_000 \
#        --learning_rate 0.0002 \
#        --grad_norm_clipping 1280 \
#        --epsilon 0.01 \
#        --entropy_cost 0.01 \
#        --unroll_length 50 --batch_size 32"]

# Docker commands:
#   docker rm torchbeast -v
#   docker build -t torchbeast .
#   docker run --name torchbeast torchbeast
# or
#   docker run --name torchbeast -it torchbeast /bin/bash
