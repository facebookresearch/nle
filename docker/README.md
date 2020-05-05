# Docker images for NLE

This directory -- i.e., `docker/` -- contains some docker images that we use for
testing both locally and in CI.

To build any of them (e.g. the main `Dockerfile`) do:

```bash
# in the repository root
$ DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile . -t nle_test
```
