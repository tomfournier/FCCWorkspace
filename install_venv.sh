#!/bin/bash
export ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# check if python3 is available locally
if ! [[ $(command -v python3) ]] 
then
    echo "Error: python3 not available locally, cannot continue!"
    exit 1
fi

echo "> getting latest pip"
if ! [[ $(command -v pip)]]
then 
    python -m pip install --upgrade pip
else
    pip install --upgrade pip

if ! [[ $(command -v pdm) ]] 
then
    echo "> installing pdm"
    pip install pdm
    echo "> Initiating pdm, select venv as environment name"
    pdm init
else
    echo "> pdm already installed, skipping"
    echo "> Initiating pdm, select venv as environment name"
    pdm init
fi

echo "> Activating environment"
eval $(pdm venv activate)
pdm config --local venv.with_pip true
echo "> Adding required packages to the environment"
pdm add seaborn uproot tqdm awkward zfit xgboost scikit-learn hyperopt hpogrid \
    graphviz pandas atlasplots
pdm export --format=requirements --without-hashes > requirements.txt
echo "> required packages are displayed in requirements.txt"