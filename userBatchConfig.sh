# Set SCRIPT_DIR to the FCCWorkspace
export SCRIPT_DIR="$(cd "$(dirname "$LOCAL_DIR")" && pwd)"

# Add analysis folders for python to detect custom modules
export PYTHONPATH=$SCRIPT_DIR/python:$PYTHONPATH
export PYTHONPATH=$SCRIPT_DIR/analysis:$PYTHONPATH
export PYTHONPATH=$SCRIPT_DIR/analysis/ZH:$PYTHONPATH
export PYTHONPATH=$SCRIPT_DIR/analysis/ZH/xsec:$PYTHONPATH
export PYTHONPATH=$SCRIPT_DIR/analysis/ZH/mass:$PYTHONPATH
export PYTHONPATH=$SCRIPT_DIR/analysis/ZH/others:$PYTHONPATH