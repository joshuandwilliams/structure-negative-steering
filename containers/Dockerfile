# =============================================================================
# negsteer runtime image: Boltz-2, its CUDA stack, and the engine.
#
#   docker build -t negsteer:dev -f containers/Dockerfile .
#   apptainer build negsteer.sif docker-daemon://negsteer:dev
#
# Build from the REPOSITORY ROOT, not from containers/, because the engine is
# copied in from bin/, negsteer/ and pyproject.toml.
#
# This is a port of containers/boltz2_negsteer.def, which is the file that
# built the image every committed number came from. Every pin and every
# ordering decision below is that file's, and the comments explaining why are
# kept rather than summarised. Two things are deliberately different:
#
#   1. The base image replaces the def's hand-rolled NVIDIA yum-repo setup,
#      because nvidia/cuda:12.8.1-base-rockylinux9 ships cuda.repo already
#      configured. It is the -base- variant, not -cudnn-runtime-: that one is
#      4.73 GB uncompressed and bundles a CUDA runtime and cuDNN which the dnf
#      list below then installs over the top, so part of it was paid for twice.
#      -base- is 0.18 GB compressed and the dnf list adds exactly what is
#      needed, including libcudnn9-cuda-12, which is what the def did.
#   2. Python comes from dnf, not from source. The def compiled 3.11.9 to get
#      sqlite support; Rocky 9 AppStream ships python3.11-3.11.13 built with
#      it, so the compile is unnecessary. The repo's floor is 3.10.
#
# WHAT THE DEF'S BUILT IMAGE CONTAINED, read back off it rather than inferred
# (2026-08-17): boltz 2.2.1, torch 2.6.0+cu124, numpy 1.26.4, pandas 2.3.3,
# gemmi 0.6.5, biopython 1.84, pyyaml 6.0.2, scipy 1.13.1, matplotlib 3.10.8.
# boltz is pinned to 2.2.1 here where the def left it floating, because that
# is the version the committed benchmark used.
#
# SIZE. The three weight files alone are 6.21 GB (conf 2.29, aff 2.06, mols
# 1.86, measured from the HuggingFace headers), and the base is 2.9 GB
# compressed. This will not build on a standard GitHub-hosted runner, which
# has 14 GB of SSD. Build it on the software node.
# =============================================================================
FROM nvidia/cuda:12.8.1-base-rockylinux9

# Fail the build on the first error in a RUN, not the last.
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# pip's wheel cache is worth 3.2 GB in this image and nothing at runtime. It
# has to be off from the start rather than deleted later, because a Docker
# layer only adds: removing a file in a later RUN leaves it in the earlier
# layer and the image does not shrink.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# ── Interpreter, build tools, and the CUDA libraries torch dlopens ─────────
#
# The CUDA packages are the def's list, installed wholesale rather than
# discovered one at a time. `import torch` dlopens several of these directly,
# and the -cudnn-runtime- base carries only some: a build with just the base
# fails on libcupti.so.12, and adding that alone then fails on
# libcusparseLt.so.0. The def had already established the complete set
# empirically, so this is that set, including libcudnn9-cuda-12, since the
# -base- image carries no cuDNN.
#
# libnpp-12-8 is the one package from the def's list left out. It is NVIDIA's
# image and signal processing library, 305 MB, and nothing here loads it:
# removing it leaves torch, boltz, trifast, cuequivariance_torch, the engine
# imports and `negsteer --version` all working. libnccl stays despite a
# single-GPU task never needing collectives, because torch._C links it and
# `import torch` fails without it.
#
# The base image ships NVIDIA's cuda.repo, so none of the def's repo setup is
# needed to reach them.
#
# python3.11-devel is needed because trifast builds from a source distribution.
RUN dnf install -y --setopt=install_weak_deps=False --setopt=obsoletes=0 \
        python3.11 python3.11-devel python3.11-pip \
        cuda-cudart-12-8 \
        cuda-nvrtc-12-8 \
        libcublas-12-8 \
        libcusparse-12-8 \
        libcusolver-12-8 \
        libcurand-12-8 \
        libcufft-12-8 \
        libnvjitlink-12-8 \
        cuda-cupti-12-8 \
        cuda-nvtx-12-8 \
        libcusparselt0 \
        libnccl \
        libcudnn9-cuda-12 \
        gcc gcc-c++ make wget tar \
 && dnf clean all \
 && rm -rf /var/cache/dnf \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && python -m pip install --upgrade pip setuptools

# ── torch, with --no-deps, which is the load-bearing decision ───────────────
# torch 2.6 declares hard == pins on nvidia-cublas-cu12==12.4.127 and friends,
# while cuequivariance-ops 0.9.1 needs cuBLAS >= 12.5 for
# cublasGemmGroupedBatchedEx. Those are irreconcilable through pip: any attempt
# to upgrade the pip nvidia wheels ends in ResolutionImpossible once pip
# re-evaluates the tree. The CUDA runtime comes from the base image instead, so
# --no-deps is what stops pip ever seeing the nvidia-*-cu12 packages at all.
# +cu124 torch is ABI-compatible with the 12.8 runtime libraries.
RUN pip install --no-deps torch==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124

# torch's own non-CUDA runtime dependencies, which --no-deps skipped. These are
# its wheel METADATA Requires-Dist entries minus the nvidia-*-cu12 ones.
# Versions left floating, because torch's floors are loose.
RUN pip install \
        filelock \
        typing-extensions \
        'sympy==1.13.1' \
        networkx \
        jinja2 \
        fsspec \
        'triton>=3.2,<3.3' \
 && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ── The constraints file every later install honours ────────────────────────
#   torch==2.6.0            the wheel is 2.6.0+cu124 and a local version
#                           matches on the public part, so pip refuses any
#                           other torch. That is the point.
#   numpy<2.0               Boltz pins it, and DockQ breaks silently on 2.x.
#   pandas>=2.2.2,<3.0      Boltz needs >=2.2.2; pandas 3.0 pulls numpy 2.x and
#                           a previous build silently shipped 3.0.2.
#   cuequivariance 0.9.1    the version that resolved and was verified
#                           importable. A bump needs a fresh pin here.
#
# There are deliberately NO nvidia-*-cu12 pins. Constraining them would
# re-introduce the resolver deadlock that --no-deps above avoids.
RUN mkdir -p /opt/build_scratch \
 && cat > /opt/build_scratch/constraints.txt << 'EOF'
torch==2.6.0
numpy<2.0
pandas>=2.2.2,<3.0
cuequivariance==0.9.1
cuequivariance-torch==0.9.1
cuequivariance-ops-cu12==0.9.1
cuequivariance-ops-torch-cu12==0.9.1
EOF

# ── cuEquivariance, which provides Boltz's CUDA kernels ─────────────────────
# The ops packages go in with --no-deps: they declare their own nvidia wheel
# requirements, and letting pip act on those is the deadlock again.
#
# Each stage then asserts torch did not drift. A silent torch replacement is
# the failure this whole ordering exists to prevent, and it surfaces at
# runtime as a kernel that will not load rather than as an install error.
RUN pip install -c /opt/build_scratch/constraints.txt \
        cuequivariance==0.9.1 \
        cuequivariance-torch==0.9.1 \
 && pip install -c /opt/build_scratch/constraints.txt --no-deps \
        cuequivariance-ops-cu12==0.9.1 \
        cuequivariance-ops-torch-cu12==0.9.1 \
 && V=$(python -c "import torch; print(torch.__version__)") \
 && case "$V" in 2.6.0+cu124|2.6.0) echo "OK torch $V preserved" ;; \
      *) echo "FATAL: cuEquivariance drifted torch to $V"; exit 1 ;; esac

# ── Boltz-2 ─────────────────────────────────────────────────────────────────
# trifast is pinned explicitly because a plain `pip install trifast` under
# these constraints makes pip backtrack through older releases looking for one
# whose dependencies it can satisfy, and the version it lands on is not the
# one that was verified.
# --no-deps on torch stops pip installing the nvidia-*-cu12 wheels, but only
# while nothing re-resolves torch's requirements. Installing anything that
# depends on torch makes pip notice they are missing and fetch all 13, at
# torch's pinned 12.4 versions, on top of the dnf CUDA 12.8 libraries above.
# That is 2.7 GB of a second CUDA runtime.
#
# They are removed in the SAME RUN rather than a later one, because a Docker
# layer only adds: a later removal leaves the bytes in this layer and the image
# does not shrink. Checked after removal: torch imports,
# torch.backends.cuda.is_built() is True, and boltz imports.
RUN pip install -c /opt/build_scratch/constraints.txt trifast==0.1.13 \
 && pip install -c /opt/build_scratch/constraints.txt boltz==2.2.1 \
 && V=$(python -c "import torch; print(torch.__version__)") \
 && case "$V" in 2.6.0+cu124|2.6.0) echo "OK torch $V preserved" ;; \
      *) echo "FATAL: Boltz drifted torch to $V"; exit 1 ;; esac \
 && pip uninstall -y -q \
        nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
        nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
        nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 \
        nvidia-nvtx-cu12 \
 && python -c "import torch, boltz; assert torch.backends.cuda.is_built()" \
 && echo "OK torch and boltz import against the system CUDA libraries"

# ── What the engine itself imports ──────────────────────────────────────────
# The def also installed matplotlib, scikit-learn, DockQ, freesasa and
# MDAnalysis for the orthogonal interface metrics. Those live in
# receptor-resurfacing-pipeline: nothing under bin/ or negsteer/ imports any of
# them, checked by grep over both trees, and removing all five leaves torch,
# boltz, the engine imports, `negsteer --version` and `compute_metrics --help`
# working. They are 146 MB plus their own dependency trees.
#
# scipy and pandas stay. pandas is boltz's own pin (>=2.2.2) and scipy comes in
# underneath it, so neither is this repo's to drop.
RUN pip install -c /opt/build_scratch/constraints.txt \
        biopython gemmi scipy pandas pyyaml

# nvidia-ml-py, NOT pynvml, and this is the fix for a real build failure
# rather than a preference. cuequivariance_ops' triton autotune cache imports
# the `pynvml` module, and nvidia-ml-py is what provides it.
#
# The `pynvml` distribution became a shim for nvidia-ml-py, and from 13.0.1 it
# ships _pynvml_redirector.pth. A .pth runs at every interpreter start,
# including inside pip's isolated build subprocess, where sys.path is minimal
# and it fails with "No module named 'json'". That broke two source builds
# here: boltz pins fairscale==0.4.13, which has no wheel, and this repo's own
# wheel. Reordering only moves the failure to the next thing that builds from
# source, so the .pth has to go rather than be worked around.
#
# Checked directly: nvidia-ml-py installs no .pth, `import pynvml` works from
# it, and fairscale==0.4.13 builds with it present. The def installed pynvml
# and built fine because in August 2026 the current release was 12.0.0, which
# shipped no redirector.
RUN pip install -c /opt/build_scratch/constraints.txt nvidia-ml-py \
 && python -c "import pynvml; print('pynvml module available')"


# ── Model weights, baked in because the cluster is airgapped ────────────────
# 6.21 GB downloaded, and about 8 GB on disk because mols.tar is kept as well
# as its extracted contents. Without them a run reaches for HuggingFace at
# predict time and fails on a node with no route out, which is why
# HF_HUB_OFFLINE is set below rather than left to chance.
#
# mols.tar MUST NOT be deleted after extracting, which cost a cluster run to
# learn. boltz/main.py:210 is `if not tar_mols.exists()` and then separately
# `if not mols.exists()`, so it tests for the TAR before it tests for the
# extracted directory. An image with mols/ but no mols.tar therefore tries to
# download the CCD on every prediction, and on a compute node that dies with
# `OSError: [Errno 101] Network is unreachable` from inside boltz, which reads
# like a network problem rather than a missing file. boltz2_negsteer.def
# extracted the tar and left it in place, which is why it never saw this.
#
# The `test -s` below turns that into a build failure rather than a run
# failure, since the two files are 1.86 GB of apparent duplication and look
# exactly like something worth deleting.
ENV BOLTZ_CACHE=/opt/boltz_cache \
    TORCH_HOME=/opt/torch_hub
RUN mkdir -p "$BOLTZ_CACHE" "$TORCH_HOME" \
 && wget -q -O "$BOLTZ_CACHE/boltz2_conf.ckpt" \
      https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt \
 && wget -q -O "$BOLTZ_CACHE/boltz2_aff.ckpt" \
      https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt \
 && wget -q -O "$BOLTZ_CACHE/mols.tar" \
      https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar \
 && tar xf "$BOLTZ_CACHE/mols.tar" -C "$BOLTZ_CACHE" \
 && chmod -R a+rX "$BOLTZ_CACHE" "$TORCH_HOME" \
 && test -s "$BOLTZ_CACHE/mols.tar" && test -d "$BOLTZ_CACHE/mols"

# ── The engine ──────────────────────────────────────────────────────────────
# bin/ is staged as a directory of standalone entry points invoked by path,
# which is how the workflow calls it, and negsteer/ is installed so `negsteer`
# is on PATH for anything calling the tool interface. --no-deps because
# everything it imports is already pinned above.
COPY negsteer      /opt/structure-negative-steering/negsteer
COPY bin           /opt/structure-negative-steering/bin
COPY pyproject.toml /opt/structure-negative-steering/pyproject.toml
RUN pip install --no-deps /opt/structure-negative-steering \
 && negsteer --version \
 && rm -rf /opt/build_scratch

# ── Runtime environment ─────────────────────────────────────────────────────
# NUMBA_CACHE_DIR and MPLCONFIGDIR point at /tmp because the image is read-only
# under Apptainer and both libraries write a cache on first import.
#
# LD_LIBRARY_PATH puts the image's CUDA 12.8 libraries first. `apptainer
# --nv` injects the host's driver libraries, and without this an older host
# CUDA runtime can win over the one this image was built against.
# libcuda.so.1 is kernel-tied and still comes from --nv, in a path this does
# not shadow.
ENV PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    NUMBA_CACHE_DIR=/tmp \
    MPLCONFIGDIR=/tmp \
    HF_HUB_OFFLINE=1

# A last check that the three things a stage actually calls are importable and
# on PATH. It cannot test the GPU: a build node has no card, so
# torch.cuda.is_available() is False here and that is expected.
RUN python -c "import boltz, torch, gemmi, Bio, yaml, numpy; print('imports OK')" \
 && command -v boltz \
 && command -v negsteer

CMD ["/bin/bash"]
