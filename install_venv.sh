#!/bin/bash
export ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# check if python3 is available locally
if ! [[ $(command -v python3) ]] 
then
    echo "Error: python3 not available locally, cannot continue!"
    exit 1
fi

echo "> getting latest pip"
pip install --upgrade pip

if ! [[ $(command -v pdm) ]] 
then
    echo "> installing pdm"
    pip install pdm
else
    echo "> pdm already installed, skipping"
    pdm init
fi

eval $(pdm venv activate)
pdm config --local venv.with_pip true
pdm add seaborn uproot tqdm awkward zfit xgboost scikit-learn hyperopt hpogrid \
    graphviz pandas atlasplots
pdm export --format=requirements --without-hashes > requirements.txt