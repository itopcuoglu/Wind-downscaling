#!/bin/bash
# Usage: bash run_debug.sh [script.py] [port]
SCRIPT=${1:-Download_HRRR_data_pointwise.py}
PORT=${2:-5678}
python -m debugpy --listen 0.0.0.0:$PORT --wait-for-client $SCRIPT
