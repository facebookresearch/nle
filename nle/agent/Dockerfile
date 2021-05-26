# -*- mode: dockerfile -*-
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
        bison \
        build-essential \
        cmake \
        curl \
        flex \
        git \
        libbz2-dev \
        ninja-build \
        software-properties-common \
        wget \
        apt-transport-https \
        ca-certificates \
        gnupg 

# Install the latest cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
RUN apt-get update && apt-get --allow-unauthenticated install -yq cmake kitware-archive-keyring 

# Install Conda
WORKDIR /opt/conda_setup
RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x miniconda.sh && \
     ./miniconda.sh -b -p /opt/conda && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Create Env, Install Torch and Keep Env active 
RUN conda init bash
RUN conda create -n nle python=3.7
RUN conda install -n nle  pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
ENV BASH_ENV ~/.bashrc
SHELL ["conda", "run", "-n", "nle", "/bin/bash" ,"-c"]
RUN python -c 'import torch'

# Install TorchBeast
WORKDIR /opt/
RUN git clone https://github.com/condnsdmatters/torchbeast.git --branch eric/experimental-port  --recursive

WORKDIR /opt/torchbeast
RUN pip install -r requirements.txt
RUN pip install ./nest
RUN python setup.py install

# Create Workspace
WORKDIR /opt/workspace
RUN pip install nle \
    hydra-core \ 
    hydra_colorlog \
    wandb \
    einops

RUN echo "conda activate nle" >> ~/.bashrc
CMD ["/bin/bash"]

# Docker commands:
#   docker build -t nle  .
#   docker run -v current_dir:/opt/workspace -it nle
