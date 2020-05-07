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

The git repository is installed inside a conda distribution, and can be found in
`/opt/nle` inside the images.

The DockerHub repository also contains pre-built images per each released
version of `nle`, with a specific template --
`fairnle/nle:<nle-version>-<type>`. The only relevant exception is
`fairnle/nle:stable`, which refers to the image built through
`docker/Dockerfile-bionic` with the latest released version of `nle` (which you
should probably be using, if you are getting started).

Currently available types are `bionic`, and `xenial`, corresponding to the
respective dockerfiles in `docker/`. Versions are in the form of `X.Y.Z`.


# Building images locally

To build any of them (e.g. the main `Dockerfile`) do:

```bash
# in the repository root
$ docker build -f docker/Dockerfile-bionic . -t nle:test-bionic
```
