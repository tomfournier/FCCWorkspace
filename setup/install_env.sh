#!/bin/bash

###############################
### SETUP CONDA ENVIRONMENT ###
###############################

export ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PARENTDIR="$(dirname "$ROOTDIR")"

ARG="$1"

# Get Python version from path
if command -v python &> /dev/null; then
    echo "> python detected, will use it to setup the environment"
    PYTHON_CMD=python
    VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "> detected Python version: $VERSION"
    echo ""
elif command -v python3 &> /dev/null; then
    echo "> python was not detected but python3 was, will use it to setup the environment"
    PYTHON_CMD=python3
    VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "> detected Python3 version: $VERSION"
    echo ""
else
    echo "> neither python nor python3 were detected"
    echo "  is python properly setup?"
    exit 1
fi

# Determine environment
if [ "$ARG" = "fccanalysis" ]; then
    echo "> you chose 'fccanalysis' settings for your environment"
    ENV_PATH="${PARENTDIR}/.envs/fccanalysis"
elif [ "$ARG" = "combined-limit" ]; then
    echo "> you chose 'combined-limit' settings for your environment"
    ENV_PATH="${PARENTDIR}/.envs/combined-limit"
else
    if [ -z "$ARG" ]; then
        echo "> you did not specify 'fccanalysis' or 'combined-limit' as argument"
        echo "  will choose default settings"
        ENV_PATH="${PARENTDIR}/.envs/default"
    else
        echo "> unknown argument: $ARG"
        echo "  available options: fccanalysis, combined-limit, or leave empty for default"
        exit 1
    fi
fi

# Upgrade pip
echo "> upgrading pip"
"$PYTHON_CMD" -m pip install --upgrade pip || exit 1

if [ -d "$ENV_PATH" ]; then
    echo "> environment at ${ENV_PATH} already exist"
    echo "  if you want to recreate this environment, remove it and re-execute this script"
    echo "  if you want to install the modules in requirements.txt, you can run:"
    echo "  ${PYTHON_CMD} -m pip install -r ${PARENTDIR}/requirements.txt"
    exit 0
fi

# Setting environment
echo "> creating environment at: $ENV_PATH"
"$PYTHON_CMD" -m venv $ENV_PATH || exit 1

# Activate environment
echo "> activating environment"
source ${ENV_PATH}/bin/activate || exit 1

# Install requirements if present
if [ -e "${PARENTDIR}/requirements.txt" ]; then
    echo "> requirements.txt is present, will download the required modules"
    echo "  you can also add modules after having activated the environment with:"
    echo "  pip install <module_names>"
    "$PYTHON_CMD" -m pip install -r "${PARENTDIR}/requirements.txt" || exit 1
fi

echo ""
echo "> environment setup complete!"
echo "> to activate this environment in the future, run:"
echo "  source ${ENV_PATH}/bin/activate"
echo "> or you can execute setup_local.sh by using this command:"
echo "  source setup/local_env.sh <env_name>"
echo "  with env_name = fccanalysis, combined-limit or default"
echo "  will use default environment if no argument provided"
echo ""
