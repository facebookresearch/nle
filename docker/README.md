# Docker images for NLE

This directory -- i.e., `docker/` -- contains some docker images that we use for
testing both locally and in CI. They contain dependencies and a pre-built
version of NLE.

You can try out the latest stable version by doing:

```bash
$ docker pull fairnle/nle:stable
$ docker run --rm -it fairnle/nle:stable python  # or bash
# Then you can simply use the nle package as normal
```

The repository also contains one image per released version of `nle`, with the
following template: `fairnle/nle:vX.Y.Z` (e.g. `fairnle/nle:v0.1.0`). The
repository is installed inside a conda distribution, and can be found in
`/opt/nle` inside the images.


# Building images locally

To build any of them (e.g. the main `Dockerfile`) do:

```bash
# in the repository root
$ DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile . -t nle_test
```
