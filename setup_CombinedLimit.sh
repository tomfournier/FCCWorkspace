#!/bin/bash

########################################
### SETUP COMBINED-LIMIT ENVIRONMENT ###
########################################

# Set LOCAL_DIR to the directory of this script
export LOCAL_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
cd "$LOCAL_DIR"

LCG_RELEASE=LCG_106 # includes ROOT 6.32, like CMSSW_14_1_0_pre4
# LCG_RELEASE=dev3/latest # includes nightly build of ROOT master, useful for development
LCG_PATH=/cvmfs/sft.cern.ch/lcg/views/$LCG_RELEASE/x86_64-el9-gcc13-opt

source $LCG_PATH/setup.sh
source $LCG_PATH/bin/thisroot.sh

# Go to CombinedLimit folder
cd HiggsAnalysis/CombinedLimit

ARG="$1"



############################
### BUILD COMBINED-LIMIT ###
############################

EXPORT_COMMAND="export CXXFLAGS="-Werror""
CMAKE_COMMAND="cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE -DUSE_VDT=FALSE .."
MAKE_COMMAND="cmake --build . -j 8"

if [ "$ARG" == "rebuild" ]; then

    rm -rf build
    mkdir build && cd build

    eval "$EXPORT_COMMAND"
    eval "$CMAKE_COMMAND"
    eval "$MAKE_COMMAND"

    cd ../

elif [ -z "$ARG" ]; then
    if [ ! -d "build" ]; then

        mkdir build && cd build

        eval "$EXPORT_COMMAND"
        eval "$CMAKE_COMMAND"
        eval "$MAKE_COMMAND"

        cd ../

    else
        echo -e "\nCombined-Limit is already built."
        echo "If you want to rebuild it, please run:"
        echo -e "source setup_CombinedLimit.sh rebuild\n"

    fi 

elif [ "$ARG" == "build" ]; then 
    if [ -d "build" ]; then 

        cd build
        eval "$MAKE_COMMAND"

        cd ../

    else 
        mkdir build && cd build

        eval "$EXPORT_COMMAND"
        eval "$CMAKE_COMMAND"
        eval "$MAKE_COMMAND"

        cd ../
    fi
fi

export PATH=$PWD/build/bin:$PATH
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/build/python:$PYTHONPATH

cd ../../

# Make alias for fast rebuilding
echo -e "\nMaking alias for fast rebuilding"
echo "If you want to compile Combined-Limit, run: build_combine"
echo -e "If you want to recompile Combined-Limit from scratch, run: rebuild_combined\n"
alias build_combine='(cd "${LOCAL_DIR}" && source setup_CombinedLimit.sh build)'
alias rebuild_combine='(cd "${LOCAL_DIR}" && source setup_CombinedLimit.sh rebuild)'

# Writting PYTHONPATH for VSCode to detect the environment
echo -e "\nWritting PYTHONPATH into env_combinedlimit.sh.txt for VSCode"
echo -e "You can remove it and comment these lines in the script if you don't need it\n"
printenv | grep -w ^PYTHONPATH > env_combinedlimit.txt
