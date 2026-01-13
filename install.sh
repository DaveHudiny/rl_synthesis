 #!/bin/bash

# multi-core compilation
# COMPILE_JOBS=$(nproc)
# single-core compilation:
export COMPILE_JOBS=8

# environment variables
PAYNT_ROOT=`pwd`
# PREREQUISITES=${PAYNT_ROOT}/prerequisites # modify this to install prerequisites outside of Paynt

# storm and stormpy dependencies
sudo apt update -qq
sudo apt install -y build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev
sudo apt install -y maven uuid-dev python3.10-dev python3.10-venv

python3.10 -m pip install --upgrade pip

# prerequisites
# mkdir -p ${PREREQUISITES}

# setup and activate python environment
# python3.10 -m venv ${PREREQUISITES}/venv
# source ${PREREQUISITES}/venv/bin/activate
pip3 install wheel
# pip3 install paynt

# paynt dependencies
sudo apt -y install graphviz
pip3 install click z3-solver psutil graphviz

# build vec_storm
cd ${PAYNT_ROOT}/VecStorm
pip install -e .

cd ${PAYNT_ROOT}

pip install jax==0.5.3
pip install tensorflow==2.15 ml-dtypes==0.2.0
pip install tf_agents==0.19.0
pip install tqdm dill matplotlib pandas seaborn networkx
pip install aalpy
pip install scikit-learn

cd ${PAYNT_ROOT}/rl_src
pip install -e .
cd ${PAYNT_ROOT}

# done
# deactivate
