FROM python:3.10.18-slim
WORKDIR /app

ENV PAYNT_ROOT=/app \
    PREREQUISITES=${PAYNT_ROOT}/prerequisites \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/app/rl_src:/app/VecStorm:/app:$PYTHONPATH"

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libboost-all-dev \
    libcln-dev \
    libgmp-dev \
    libginac-dev \
    automake \
    libglpk-dev \
    libhwloc-dev \
    libz3-dev \
    libxerces-c-dev \
    libeigen3-dev \
    maven \
    uuid-dev \
    graphviz \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${PAYNT_ROOT}/prerequisites

# RUN python3 -m venv ${PAYNT_ROOT}/prerequisites/venv
# ENV PATH="${PAYNT_ROOT}/prerequisites/venv/bin:$PATH"



RUN pip install --upgrade pip && \
    pip install wheel

RUN pip install paynt click z3-solver psutil graphviz && \
    pip install jax==0.5.3 chex && \
    pip install tensorflow==2.15 ml-dtypes==0.2.0 && \
    pip install tf-agents==0.19.0 tqdm dill matplotlib pandas seaborn networkx && \
    pip install aalpy scikit-learn

COPY . .

RUN cd /app/VecStorm \
    pip install -e . \
    cd /app/rl_src \
    pip install -e .

RUN pip install tf-agents==0.19.0
CMD ["bash"]
