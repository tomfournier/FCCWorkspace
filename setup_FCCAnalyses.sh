#!/bin/bash

#####################################
### SETUP FCCANALYSES ENVIRONMENT ###
#####################################

# Set LOCAL_DIR to the directory of this script
export LOCAL_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
cd "$LOCAL_DIR"

# Source Key4hep and setup script
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-03-10
cd FCCAnalyses
source ./setup.sh

ARG="$1"



#########################
### BUILD FCCANALYSES ###
#########################

EXPORT_COMMAND="export CXXFLAGS="-Werror""
CMAKE_COMMAND="cmake -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE .."
MAKE_COMMAND="make -j8 install"

if [ "$ARG" == "rebuild" ]; then

    rm -rf build install
    mkdir install build && cd build

    eval "$EXPORT_COMMAND"
    eval "$CMAKE_COMMAND"
    eval "$MAKE_COMMAND"

    cd ../

elif [ -z "$ARG" ]; then
    if [ ! -d "build" ]; then

        mkdir install build && cd build

        eval "$EXPORT_COMMAND"
        eval "$CMAKE_COMMAND"
        eval "$MAKE_COMMAND"

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
        eval "$MAKE_COMMAND"

        cd ../

    else 
        mkdir install build && cd build

        eval "$EXPORT_COMMAND"
        eval "$CMAKE_COMMAND"
        eval "$MAKE_COMMAND"

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
printenv | grep -w ^PYTHONPATH > env_fccanalysis.txt
