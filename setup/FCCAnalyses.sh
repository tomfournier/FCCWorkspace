#!/bin/bash

#####################################
### SETUP FCCANALYSES ENVIRONMENT ###
#####################################

# Set LOCAL_DIR to the FCCWorkspace
export ROOTDIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
export LOCAL_DIR="$(dirname "$ROOTDIR")"
cd "$LOCAL_DIR"

# Source Key4hep and setup script
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-03-10
cd FCCAnalyses
source ./setup.sh

ARG="$1"



#########################
### BUILD FCCANALYSES ###
#########################

EXPORT_CMD="export CXXFLAGS="-Werror""
CMAKE_CMD="cmake -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE .."
MAKE_CMD="make -j8 install"

if [ "$ARG" == "rebuild" ]; then

    rm -rf build install
    mkdir install build && cd build

    eval "$EXPORT_CMD"
    eval "$CMAKE_CMD"
    eval "$MAKE_CMD"

    cd ../

elif [ -z "$ARG" ]; then
    if [ ! -d "build" ]; then

        mkdir install build && cd build

        eval "$EXPORT_CMD"
        eval "$CMAKE_CMD"
        eval "$MAKE_CMD"

        cd ../

    else
        echo -e "\nFCCAnalyses is already built."
        echo "If you want to rebuild it, please run:"
        echo -e "source setup_FCCAnalyses.sh rebuild\n"

        echo "You can also run the following command if you want the default setup:"
        echo "cd FCCAnalyses"
        echo "source ./setup.sh"
        echo "fccanalysis build -j 8"
        echo -e "cd ..\n"
    fi 

elif [ "$ARG" == "build" ]; then 
    if [ -d "build" ]; then 

        cd build
        eval "$MAKE_CMD"

        cd ../

    else 
        mkdir install build && cd build

        eval "$EXPORT_CMD"
        eval "$CMAKE_CMD"
        eval "$MAKE_CMD"

        cd ../
    fi
fi

cd ../

# Make alias for fast rebuilding
echo -e "\nMaking alias for fast rebuilding"
echo "If you want to compile FCCAnalyses, run: build_fcc"
echo -e "If you want to recompile FCCAnalyses from scratch, run: rebuild_fcc\n"
alias build_fcc='(cd "${LOCAL_DIR}" && source setup_FCCAnalyses.sh build)'
alias rebuild_fcc='(cd "${LOCAL_DIR}" && source setup_FCCAnalyses.sh rebuild)'

echo "You can also run 'fccanalysis build -j8' for default compiling"

# Writting PYTHONPATH for VSCode to detect the environment
echo -e "\nWritting PYTHONPATH into env_fccanalysis.txt for VSCode"
echo -e "You can remove it and comment these lines in the script if you don't need it\n"
if [ ! -d "envs" ]; then
    mkdir envs
fi
printenv | grep -w ^PYTHONPATH > envs/fccanalysis.txt
