#!/bin/bash
# quick rebuild script for all neurova c++ modules
# platform: macos arm64 (apple silicon)

set -e  # exit on error

echo "========================================="
echo "neurova c++ module compilation"
echo "========================================="

# configuration
PYTHON_VERSION="3.12"
VENV_PATH="/Users/harrythapa/Desktop/nalyst-research/.venv"
PYBIND_INCLUDE="$VENV_PATH/lib/python$PYTHON_VERSION/site-packages/pybind11/include"
PYTHON_INCLUDE="/opt/homebrew/opt/python@$PYTHON_VERSION/Frameworks/Python.framework/Versions/$PYTHON_VERSION/include/python$PYTHON_VERSION"

# compiler flags
CXXFLAGS="-O3 -march=armv8-a -std=c++17 -shared -fPIC -undefined dynamic_lookup"
INCLUDES="-I$PYBIND_INCLUDE -I$PYTHON_INCLUDE"

# output directory
OUT_DIR="neurova"
SRC_DIR="src"

# modules to compile
MODULES=(
    "neurova_minimal"
    "neurova_architecture"
    "neurova_extended"
    "neurova_mega"
    "neurova_advanced"
    "neurova_final"
    "neurova_timeseries"
)

# compile each module
for module in "${MODULES[@]}"; do
    echo ""
    echo "compiling $module..."
    clang++ $CXXFLAGS $INCLUDES \
        "$SRC_DIR/$module.cpp" \
        -o "$OUT_DIR/$module.cpython-312-darwin.so"
    
    if [ $? -eq 0 ]; then
        size=$(ls -lh "$OUT_DIR/$module.cpython-312-darwin.so" | awk '{print $5}')
        echo "compiled $module successfully ($size)"
    else
        echo "compilation failed for $module"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "compilation complete"
echo "========================================="
echo ""

# summary
total_size=$(du -sh $OUT_DIR/*.so | tail -1 | awk '{print $1}')
num_modules=$(ls -1 $OUT_DIR/*.so | wc -l | tr -d ' ')

echo "total modules: $num_modules"
echo "total size: $total_size"
echo ""

# verify
echo "verifying modules..."
$VENV_PATH/bin/python verify_cpp.py

echo ""
echo "all done"
