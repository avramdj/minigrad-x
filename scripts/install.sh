#!/bin/bash

set -e

if [ "$1" == "--no-cache" ]; then
    CACHE_ARGS="--no-cache-dir"
    pip uninstall -y minigradx
fi

pip install $CACHE_ARGS -e .[dev,cuda]
