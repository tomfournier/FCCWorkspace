#!/bin/bash
export ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "> getting latest pip"
python3 -m pip install --upgrade pip

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
pdm install
pdm export --format=requirements --without-hashes > requirements.txt
echo "> required packages are displayed in requirements.txt"