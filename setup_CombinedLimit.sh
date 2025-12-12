#!/bin/bash

# Go to CombinedLimit folder
cd HiggsAnalysis/CombinedLimit

# Check if the build directory exists
if [ ! -d "build" ]; then
    make -j 4
else
    echo "CombinedLimit is already built."
fi

source env_standalone.sh

cd ../../
