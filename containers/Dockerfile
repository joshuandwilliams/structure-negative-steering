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
#   1. The base image replaces 123 lines of hand-rolled NVIDIA yum-repo setup.
#      nvidia/cuda:12.8.1-cudnn-runtime-rockylinux9 is the same RHEL 9 family
#      as the def's AlmaLinux 9, with CUDA 12.8 runtime and cuDNN 9 already in
#      it, which is exactly what that section installed by hand.
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
FROM nvidia/cuda:12.8.1-cudnn-runtime-rockylinux9

# Fail the build on the first error in a RUN, not the last.
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# ── Interpreter and build tools ──────────────────────────────────────────────
# python3.11-devel is needed because trifast builds from a source distribution.
RUN dnf install -y --setopt=install_weak_deps=False \
        python3.11 python3.11-devel python3.11-pip \
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

# pynvml is required by cuequivariance_ops' triton autotune cache.
RUN pip install -c /opt/build_scratch/constraints.txt pynvml

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
RUN pip install -c /opt/build_scratch/constraints.txt trifast==0.1.13 \
 && pip install -c /opt/build_scratch/constraints.txt boltz==2.2.1 \
 && V=$(python -c "import torch; print(torch.__version__)") \
 && case "$V" in 2.6.0+cu124|2.6.0) echo "OK torch $V preserved" ;; \
      *) echo "FATAL: Boltz drifted torch to $V"; exit 1 ;; esac

# ── What the engine itself imports, plus the orthogonal metrics ────────────
RUN pip install -c /opt/build_scratch/constraints.txt \
        matplotlib biopython gemmi scipy scikit-learn pandas pyyaml \
 && pip install -c /opt/build_scratch/constraints.txt DockQ freesasa MDAnalysis

# ── Model weights, baked in because the cluster is airgapped ────────────────
# 6.21 GB in three files. Without them a run reaches for HuggingFace at
# predict time and fails on a node with no route out, which is why
# HF_HUB_OFFLINE is set below rather than left to chance.
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
 && rm -f "$BOLTZ_CACHE/mols.tar" \
 && chmod -R a+rX "$BOLTZ_CACHE" "$TORCH_HOME"

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
