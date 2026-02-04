#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Comprehensive Neurova Test Suite"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("NEUROVA COMPREHENSIVE TEST SUMMARY")
print("=" * 60)

tests = [
    (['pytest', 'tests/', '-q', '--tb=no'], 'Unit Tests (pytest)'),
    (['test_nvc.py'], 'NVC Core'),
    (['test_ml.py'], 'ML Module'),
    (['test_neural.py'], 'Neural Networks'),
    (['test_nvc_filters.py'], 'CV Filters'),
    (['test_new_features.py'], 'New Features'),
    (['test_new_modules.py'], 'New Modules'),
    (['test_advanced.py'], 'Advanced Features'),
    (['test_architecture.py'], 'Architectures'),
    (['test_augmentation.py'], 'Augmentation'),
    (['test_no_cv2.py'], 'No OpenCV Required'),
]

passed = 0
failed = 0
results = []

for test_args, name in tests:
    if test_args[0] == 'pytest':
        cmd = [sys.executable, '-m'] + test_args
    else:
        cmd = [sys.executable] + test_args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"   {name}")
            passed += 1
            results.append((name, "PASS"))
        else:
            print(f"   {name}")
            failed += 1
            results.append((name, "FAIL"))
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {name} (timeout)")
        failed += 1
        results.append((name, "TIMEOUT"))
    except Exception as e:
        print(f"   {name}: {e}")
        failed += 1
        results.append((name, "ERROR"))

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"  PASSED: {passed}")
print(f"  FAILED: {failed}")
print(f"  TOTAL:  {passed + failed}")
print("=" * 60)

if failed == 0:
    print("\n ALL TESTS PASSED!")
    print("\nNeurova is ready for PyPI publication.")
else:
    print(f"\n  {failed} test suite(s) failed")
    
sys.exit(0 if failed == 0 else 1)
