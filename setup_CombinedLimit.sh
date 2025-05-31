#!/bin/bash

# Source CombinedLimit
cd HiggsAnalysis/CombinedLimit
. env_standalone.sh

# Check if the build directory exists
if [ ! -d "build" ]; then
    make -j 4
else
    echo "CombinedLimit is already built."
    echo "If you want to rebuild it, please run:"
    echo "cd HiggsAnalysis/CombinedLimit"
    echo ". env_standalone.sh"
    echo "cd ../../"

cd ../../