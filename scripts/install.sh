#!/bin/bash

git submodule update --init --recursive

# Cleans up any previous installations of our library
rm -f minigradx/_C.pyi

set -e

if [ "$1" == "--no-cache" ]; then
    CACHE_ARGS="--no-cache-dir"
    pip uninstall -y minigradx
fi

rm minigradx/_C.pyi minigradx/_C*.so || true
pip install $CACHE_ARGS -e .[dev,cuda]
python setup.py generate_stubs -v
