FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    libtool \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-numpy \
    git \
    cmake \
    libzmqpp-dev \
    libncurses5-dev

WORKDIR /src

RUN git clone https://github.com/google/flatbuffers.git

WORKDIR /src/flatbuffers

RUN cmake -G "Unix Makefiles"

RUN make

RUN make install

WORKDIR /nethack

COPY . /nethack

RUN make clean

WORKDIR /nethack/sys/unix

RUN sh setup.sh hints/linux

WORKDIR /nethack

RUN make

RUN make install

RUN pip3 install zmq

RUN pip3 install flatbuffers

# CMD ["server"]

# Docker commands:
#   docker rm nethack -v
#   docker build -t nethack .
#   docker run --name nethack nethack
# or
#   docker run --name nethack -it nethack /bin/bash
