# -*- mode: dockerfile -*-

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG PYTHON_VERSION=3.8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
    apt-transport-https \
    ca-certificates \
    gnupg \
    wget \
    git \
    software-properties-common \
    build-essential \
    emacs \
    vim \
    python3-pip

RUN apt-get update && apt-get install -y \
    curl \
    make \
    g++ \
    clang \
    bison \
    flex \
    libncurses-dev \
    libzmq3-dev

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -

RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get --allow-unauthenticated install -yq \
    cmake \
    kitware-archive-keyring

WORKDIR /opt/flatbuffers

RUN curl -fSL "https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz" -o flatbuffers.tar.gz \
    && tar xzf flatbuffers.tar.gz \
    && cd flatbuffers-* \
    && cmake \
       -G "Unix Makefiles" \
       -DCMAKE_BUILD_TYPE=Release \
    && make \
    && make install \
    && cp src/idl_parser.cpp src/idl_gen_text.cpp /usr/local/include/flatbuffers


WORKDIR /opt/conda_setup

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x miniconda.sh && \
     ./miniconda.sh -b -p /opt/conda && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN python -m pip install --upgrade pip ipython ipdb

WORKDIR /opt/nle
COPY . .

RUN pip install ".[all]"

WORKDIR '/workspace'
