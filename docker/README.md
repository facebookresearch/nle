# Docker images for NLE

This directory -- i.e., `docker/` -- contains some docker images that we use for
testing both locally and in CI. They contain dependencies and a pre-built
version of NLE.

You can try out the latest stable version in Ubuntu 18.04 by doing:

```bash
$ docker pull fairnle/nle:stable
$ docker run --gpus all --rm -it fairnle/nle:stable python  # or bash
# Then you can simply use the nle package as normal
```

The git repository is installed inside a conda distribution, and can be found in
`/opt/nle` inside the images.

The DockerHub repository also contains pre-built images per each released
version of `nle`, following a specific templates:

``` bash
1.  fairnle/nle:stable
2.  fairnle/nle:<nle-version>         # corresponds to (1), based on Ubuntu 18.04
3.  fairnle/nle-xenial:<nle-version>  # Based on Ubuntu 16.04
4.  fairnle/nle-focal:<nle-version>   # Based on Ubuntu 20.04
5.  fairnle/nle:<sha>                 # bionic image built on dockerfile changes
6.  fairnle/nle-xenial:<sha>          # xenial image built on dockerfile changes
7.  fairnle/nle-focal:<sha>           # focal image built on dockerfile changes
8.  fairnle/nle:dev                   # points to latest built sha
9.  fairnle/nle-xenial:dev            # points to latest built sha
10. fairnle/nle-focal:dev             # points to latest built sha
```

`<nle-version>` is the latest pip version released, and follows semantic versioning (so something like `X.Y.Z`).

# Building images locally

To build and run any of them (e.g. `Dockerfile-bionic`) do:

```bash
$ git clone https://github.com/facebookresearch/nle --recursive
$ cd nle
$ docker build -f docker/Dockerfile-bionic . -t nle
$ docker run --gpus all --rm --name nle nle
# or alternatively
$ docker run --gpus all -it --entrypoint /bin/bash nle
```
