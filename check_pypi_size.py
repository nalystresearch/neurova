#!/usr/bin/env python3
"""Check package size for PyPI compatibility."""

import os

data_size = 0
code_size = 0

for root, dirs, files in os.walk('neurova'):
    dirs[:] = [d for d in dirs if d != '__pycache__']
    for f in files:
        if f.endswith('.pyc') or f.endswith('.pyo'):
            continue
        fp = os.path.join(root, f)
        size = os.path.getsize(fp)
        if '/data/' in fp:
            data_size += size
        else:
            code_size += size

print("=" * 60)
print("NEUROVA PACKAGE SIZE ANALYSIS")
print("=" * 60)
print(f"Code files: {code_size / 1024 / 1024:.2f} MB")
print(f"Data files: {data_size / 1024 / 1024:.2f} MB")
print(f"Total package: {(code_size + data_size) / 1024 / 1024:.2f} MB")
print()
print("PyPI Limits:")
print("  Source distribution: 100 MB (LIMIT)")
print("  Wheel: 100 MB (LIMIT)")
print()

total = code_size + data_size
if total < 100 * 1024 * 1024:
    print("STATUS: COMPATIBLE FOR PYPI UPLOAD")
else:
    print("STATUS: TOO LARGE - NEED TO REDUCE SIZE")
    print()
    print("Recommendations:")
    print("  - Remove large data files from package")
    print("  - Host data files separately (e.g., GitHub releases)")
    print("  - Use lazy downloading for datasets")
