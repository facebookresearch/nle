# -*- mode: dockerfile -*-

FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
        bison \
        build-essential \
        cmake \
        flex \
        libncurses5-dev \
        ninja-build

COPY . /opt/nethack/

RUN rm -rf /opt/nethack/build

WORKDIR /opt/nethack/build

RUN cmake .. -GNinja && ninja && cmake --install .

CMD ["./nethack"]


# Docker commands:
#   docker rm nethack -v
#   docker build -t nethack .
#   docker run -it --rm --name nethack nethack
# or
#   docker run -it --entrypoint /bin/bash nethack
