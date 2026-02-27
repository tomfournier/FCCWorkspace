export ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PARENTDIR="$(dirname "$ROOTDIR")"

ARG="$1"

# Setting environment name
if [ "$ARG" = "fccanalysis" ]; then
    ENV_NAME="fccanalysis"
elif [ "$ARG" = "combined-limit" ]; then
    ENV_NAME="combined-limit"
elif [ -z "$ARG" ] || [ "$ARG" = "default" ]; then
    ENV_NAME="default"
else
    echo "> unknown argument: $ARG"
    echo "  available options: fccanalysis, combined-limit, default or leave empty for default"
    exit 1
fi



# Checking environment existence
ENV_PATH=${PARENTDIR}/.envs/${ENV_NAME}
echo "> activating ${ENV_NAME} environment at ${ENV_PATH}"
if [ -d "$ENV_PATH" ]; then
    source "${ENV_PATH}/bin/activate"
else
    echo "> There is no environment at ${ENV_PATH}"
    echo "  Be sure to source ${ENV_NAME} environment by sourcing install_env.sh before sourcing this script"
    exit 1
fi
