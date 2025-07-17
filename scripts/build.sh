#!/usr/bin/env bash
set -euo pipefail

# root of your repo
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
BUILD_DIR="$PROJECT_ROOT/build"

pushd "$PROJECT_ROOT" >/dev/null

# clean + rebuild
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$PROJECT_ROOT"
cmake --build .
cd "$PROJECT_ROOT"

# find every .so, generate module path, then stubgen
find "$BUILD_DIR" -name "*.so" | while read -r so; do
  # strip build/ prefix, drop .so, convert slashes to dots
  rel="${so#$BUILD_DIR/}"
  mod="${rel%.so}"
  mod="${mod//\//.}"

  echo "Generating stubs for: $mod"
  PYTHONPATH="$BUILD_DIR" .venv/bin/stubgen -p "$mod" -o .
done

popd >/dev/null
