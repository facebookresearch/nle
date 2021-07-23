# Docker images for NLE

This directory -- i.e., `docker/` -- contains some Docker images that we use for
testing both locally and in CI. They contain dependencies and a pre-built
version of NLE.

You can try out the latest stable version in Ubuntu 18.04 by doing:

```bash
$ docker pull fairnle/nle
$ docker run --rm -it fairnle/nle python -m nle.scripts.play -e NetHackScore-v0
```

The git repository is installed inside a conda distribution, and can be found in
`/opt/nle` inside the images.

The DockerHub repository also contains pre-built images per each released
version of `nle`, following a specific templates:

``` bash
3.  fairnle/nle-xenial:<nle-version>  # based on Ubuntu 16.04, CUDA 10.2, cuDNN 7
3.  fairnle/nle-bionic:<nle-version>  # based on Ubuntu 18.04, CUDA 10.2, cuDNN 7
4.  fairnle/nle-focal:<nle-version>   # based on Ubuntu 20.04, CUDA 11.0, cuDNN 8
3.  fairnle/nle-xenial:latest         # points to latest built version
3.  fairnle/nle-bionic:latest         # points to latest built version
4.  fairnle/nle-focal:latest          # points to latest built version
6.  fairnle/nle-xenial:<sha>
6.  fairnle/nle-bionic:<sha>
7.  fairnle/nle-focal:<sha>
9.  fairnle/nle-xenial:dev            # points to latest built sha
8.  fairnle/nle-bionic:dev            # points to latest built sha
10. fairnle/nle-focal:dev             # points to latest built sha
```

`<nle-version>` is the latest pip version released, and follows semantic versioning (so something like `X.Y.Z`).

# Building images locally

To build any of them (e.g. `Dockerfile-bionic`) do:

```bash
$ git clone https://github.com/facebookresearch/nle --recursive
$ cd nle
$ docker build -f docker/Dockerfile-bionic . -t nle:latest
```

# Using prebuilt images 

You can use any prebuilt images as a base for your own Dockerfiles in the usual way:

```
FROM fairnle/nle-bionic  # Ubuntu 18.04

RUN [... etc ]
```
