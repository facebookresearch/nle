FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq git build-essential python3-pip

RUN apt-get update && apt-get install -y \
    ca-certificates \
    cmake \
    curl \
    make \
    g++ \
    clang \
    bison \
    flex \
    libzmq3-dev

RUN curl -fSL "https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz" -o flatbuffers.tar.gz \
    && tar xzf flatbuffers.tar.gz \
    && cd flatbuffers-* \
    && cmake \
       -G "Unix Makefiles" \
       -DCMAKE_BUILD_TYPE=Release \
    && make \
    && make test \
    && make install \
    && cp src/idl_parser.cpp src/idl_gen_text.cpp /usr/local/include/flatbuffers

RUN python3 -m pip install --upgrade pip

COPY . /nle

RUN cd /nle && pip3 install -e ".[all]"

WORKDIR '/nle'
