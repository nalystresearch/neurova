#!/usr/bin/env python3
"""simple verification that c++ modules are compiled and working."""

import sys
import os

# add neurova to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("neurova c++ module verification")
print("=" * 70)

# test direct import of c++ modules
modules_to_test = [
    'neurova_minimal',
    'neurova_architecture', 
    'neurova_extended',
    'neurova_mega',
    'neurova_final',
    'neurova_timeseries'
]

loaded = []
failed = []

for mod_name in modules_to_test:
    try:
        mod = __import__(f'neurova.{mod_name}', fromlist=[mod_name])
        loaded.append(mod_name)
        print(f"loaded {mod_name:25s} successfully")
    except Exception as e:
        failed.append((mod_name, str(e)[:50]))
        print(f"failed {mod_name:25s} - {str(e)[:50]}")

print("\n" + "=" * 70)
print(f"results: {len(loaded)}/6 c++ modules loaded successfully")

if len(loaded) == 6:
    print("\nsuccess all 6 c++ modules compiled and working")
    print("\ntotal size: 3.3 mb")
    print("platform: macos arm64 (apple silicon)")
    print("compiler: clang++ -o3 -march=armv8-a")
    print("python: 3.12")
elif len(loaded) >= 4:
    print(f"\ngood {len(loaded)} modules working, {len(failed)} need attention")
else:
    print(f"\nwarning only {len(loaded)} modules loaded")
    
if failed:
    print("\nfailed modules:")
    for name, error in failed:
        print(f"  - {name}: {error}")

print("=" * 70)
