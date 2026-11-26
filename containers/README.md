# containers

Two images, because one runs the engine and one tests it.

| File | Image | Contents |
|:---|:---|:---|
| `Dockerfile` | runtime | Boltz-2 2.2.1, torch 2.6.0+cu124, the CUDA 12.8 stack, the Boltz weights, and the engine |
| `Dockerfile.test` | test | Python 3.11, numpy, gemmi, biopython, PyYAML, pytest. No torch, no CUDA, no Boltz |
| `boltz2_negsteer.def` | runtime | The Apptainer definition the runtime `Dockerfile` was ported from. See below |

## Building

Build from the repository root, not from here, because the runtime image copies
in `bin/`, `negsteer/` and `pyproject.toml`.

```bash
docker build -t negsteer:dev      -f containers/Dockerfile      .
docker build -t negsteer-test:dev -f containers/Dockerfile.test .
```

Apptainer converts either one without a registry:

```bash
apptainer build negsteer.sif      docker-daemon://negsteer:dev
apptainer build negsteer-test.sif docker-daemon://negsteer-test:dev
```

Then point the pipeline at it with `NEGSTEER_BOLTZ_IMG` in `site.conf`, and the
`check_test_run` assertions at the test image with `NEGSTEER_TEST_CONTAINER`.

The runtime image will not build on a GitHub-hosted runner. The three weight
files are 6.21 GB on their own (`boltz2_conf.ckpt` 2.29, `boltz2_aff.ckpt`
2.06, `mols.tar` 1.86, from the HuggingFace response headers) and the CUDA base
is another 2.9 GB compressed, against the 14 GB of SSD a standard runner has.
Build it on a machine with disk and a route out to the internet.

## Why the weights are inside the image

The cluster's compute nodes have no route to the internet, so a Boltz call that
reached for HuggingFace at predict time would fail there rather than at build
time. The runtime image therefore bakes the weights in and sets
`HF_HUB_OFFLINE=1`, which turns a missing-weight bug into a build failure on a
machine that can still fix it.

## If a build fails on a certificate

NBI intercepts TLS. Inside a container the system trust store is the image's
own, so an https fetch can fail with `self-signed certificate in certificate
chain` even though the same URL works on the host. This is recorded here
because the symptom reads like a dead package index rather than a trust
problem.

In practice it bit conda, not pip: `JoshW_Software/RoseTTAFold-All-Atom/rfaa.def`
documents `conda.anaconda.org` returning a certificate issued by NBI's own
`nr4-CA` root, and installs that root in `%post` before any fetch. Neither
Dockerfile here uses conda, and the def these were ported from installed
everything from PyPI and `download.pytorch.org` with no certificate step and
built successfully, so none is included.

If one is ever needed, the RHEL form differs from the Debian form in that def:
drop the PEM in `/etc/pki/ca-trust/source/anchors/` and run `update-ca-trust`,
rather than `/usr/local/share/ca-certificates/` and `update-ca-certificates`.

## boltz2_negsteer.def

This is the Apptainer definition that built the image every committed number
came from, and `Dockerfile` is a port of it. It is kept until the Docker build
is verified on the software node, so there is a known-good fallback, and should
be deleted once that is done rather than maintained in parallel.

Two things the port changes deliberately:

- The base image. `nvidia/cuda:12.8.1-cudnn-runtime-rockylinux9` is the same
  RHEL 9 family as the def's AlmaLinux 9, with CUDA 12.8 and cuDNN 9 already
  installed, which replaces 123 lines of hand-rolled NVIDIA yum-repo setup and
  package selection.
- Python. The def compiled 3.11.9 from source to get sqlite support. Rocky 9
  AppStream ships `python3.11-3.11.13` built with it, so the compile is gone.

Everything else is the def's: `--no-deps` on torch, the constraints file, the
cuEquivariance and trifast pins, and the torch-drift assertions after each
stage. `boltz` is pinned to 2.2.1 where the def left it floating, because that
is the version the def's built image actually carried and the one the benchmark
used.

`boltz2_negsteer_testing.def` is gitignored and local-only. It existed to add
pytest to a copy of the runtime image, which is what `Dockerfile.test` now
does, except for one case it cannot cover: measuring the engine's own coverage
needs `coverage` installed in the process that runs the engine, so that needs
`coverage` added to the runtime image rather than a separate test image.
