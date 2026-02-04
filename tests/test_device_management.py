# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Test GPU/CPU device management for Neurova.

This test verifies that device selection works correctly across all modules.
"""

import numpy as np
import neurova as nv


def test_device_detection():
    """Test GPU detection and device info."""
    print("\n" + "="*70)
    print("TEST 1: Device Detection")
    print("="*70)
    
    # check if CUDA is available
    cuda_available = nv.cuda_is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # get device count
        gpu_count = nv.get_device_count()
        print(f"GPU count: {gpu_count}")
        
        # get GPU name
        gpu_name = nv.get_device_name()
        print(f"GPU name: {gpu_name}")
        
        # get detailed info
        info = nv.get_device_info()
        print(f"\nDevice Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        assert gpu_count > 0, "GPU detected but count is 0"
        assert len(gpu_name) > 0, "GPU name is empty"
    else:
        print("No GPU detected - skipping GPU-specific tests")
    
    print(" Device detection test passed")


def test_device_switching():
    """Test switching between CPU and GPU."""
    print("\n" + "="*70)
    print("TEST 2: Device Switching")
    print("="*70)
    
    # test CPU mode
    nv.set_device('cpu')
    assert nv.get_device() == 'cpu', "Failed to set CPU device"
    print(f" CPU mode: {nv.get_device()}")
    
    # test GPU mode (if available)
    if nv.cuda_is_available():
        nv.set_device('cuda')
        assert nv.get_device() == 'cuda', "Failed to set CUDA device"
        print(f" GPU mode: {nv.get_device()}")
        
        # test 'gpu' alias
        nv.set_device('cpu')
        nv.set_device('gpu')
        assert nv.get_device() == 'cuda', "Failed to set GPU device using 'gpu'"
        print(f" GPU alias works: {nv.get_device()}")
    else:
        print("  No GPU - skipping GPU switching test")
    
    print(" Device switching test passed")


def test_device_context():
    """Test device context manager."""
    print("\n" + "="*70)
    print("TEST 3: Device Context Manager")
    print("="*70)
    
    # set to CPU
    nv.set_device('cpu')
    assert nv.get_device() == 'cpu'
    print(f"Initial device: {nv.get_device()}")
    
    if nv.cuda_is_available():
        # use GPU context
        with nv.device_context('cuda'):
            assert nv.get_device() == 'cuda', "Context didn't switch to GPU"
            print(f"Inside context: {nv.get_device()}")
        
        # should be back to CPU
        assert nv.get_device() == 'cpu', "Context didn't restore CPU"
        print(f"After context: {nv.get_device()}")
        print(" Context manager works correctly")
    else:
        print("  No GPU - skipping context manager test")
    
    print(" Device context test passed")


def test_array_creation():
    """Test array creation on different devices."""
    print("\n" + "="*70)
    print("TEST 4: Array Creation")
    print("="*70)
    
    # cPU arrays
    nv.set_device('cpu')
    cpu_array = nv.array([1, 2, 3, 4, 5])
    cpu_zeros = nv.zeros((10, 10))
    cpu_ones = nv.ones((5, 5))
    
    print(f"CPU array type: {type(cpu_array).__module__}")
    assert 'numpy' in type(cpu_array).__module__, "CPU array not NumPy"
    print(" CPU arrays created successfully")
    
    # gPU arrays (if available)
    if nv.cuda_is_available():
        nv.set_device('cuda')
        gpu_array = nv.array([1, 2, 3, 4, 5])
        gpu_zeros = nv.zeros((10, 10))
        gpu_ones = nv.ones((5, 5))
        
        print(f"GPU array type: {type(gpu_array).__module__}")
        assert 'cupy' in type(gpu_array).__module__, "GPU array not CuPy"
        print(" GPU arrays created successfully")
    else:
        print("  No GPU - skipping GPU array creation test")
    
    print(" Array creation test passed")


def test_array_transfer():
    """Test transferring arrays between devices."""
    print("\n" + "="*70)
    print("TEST 5: Array Transfer")
    print("="*70)
    
    # create CPU array
    cpu_array = np.array([1, 2, 3, 4, 5])
    print(f"Original array type: {type(cpu_array).__module__}")
    
    if nv.cuda_is_available():
        # transfer to GPU
        gpu_array = nv.to_device(cpu_array, device='cuda')
        print(f"GPU array type: {type(gpu_array).__module__}")
        assert 'cupy' in type(gpu_array).__module__, "Failed to transfer to GPU"
        
        # transfer back to CPU
        cpu_array_2 = nv.to_device(gpu_array, device='cpu')
        print(f"Back to CPU type: {type(cpu_array_2).__module__}")
        assert 'numpy' in type(cpu_array_2).__module__, "Failed to transfer to CPU"
        
        # verify data integrity
        assert np.allclose(cpu_array, cpu_array_2), "Data corrupted during transfer"
        print(" Data integrity maintained during transfer")
    else:
        print("  No GPU - skipping transfer test")
    
    print(" Array transfer test passed")


def test_backend_selection():
    """Test backend selection (NumPy vs CuPy)."""
    print("\n" + "="*70)
    print("TEST 6: Backend Selection")
    print("="*70)
    
    # cPU backend should be NumPy
    nv.set_device('cpu')
    backend = nv.get_backend()
    print(f"CPU backend: {backend.__name__}")
    assert backend.__name__ == 'numpy', "CPU backend is not NumPy"
    print(" CPU backend is NumPy")
    
    if nv.cuda_is_available():
        # gPU backend should be CuPy
        nv.set_device('cuda')
        backend = nv.get_backend()
        print(f"GPU backend: {backend.__name__}")
        assert backend.__name__ == 'cupy', "GPU backend is not CuPy"
        print(" GPU backend is CuPy")
    else:
        print("  No GPU - skipping GPU backend test")
    
    print(" Backend selection test passed")


def test_memory_management():
    """Test GPU memory management."""
    print("\n" + "="*70)
    print("TEST 7: Memory Management")
    print("="*70)
    
    if not nv.cuda_is_available():
        print("  No GPU - skipping memory management test")
        print(" Memory management test skipped")
        return
    
    nv.set_device('cuda')
    
    # get initial memory usage
    memory_before = nv.get_memory_usage()
    print(f"Memory before: {memory_before['used_gb']:.2f} GB")
    
    # allocate some GPU memory
    large_arrays = [nv.zeros((1000, 1000)) for _ in range(10)]
    memory_during = nv.get_memory_usage()
    print(f"Memory during: {memory_during['used_gb']:.2f} GB")
    
    # clear cache
    del large_arrays
    nv.empty_cache()
    memory_after = nv.get_memory_usage()
    print(f"Memory after cleanup: {memory_after['used_gb']:.2f} GB")
    
    # memory should be freed (or at least not increased significantly)
    assert memory_after['used_gb'] <= memory_during['used_gb'], "Memory not freed"
    print(" Memory management works")
    
    print(" Memory management test passed")


def test_synchronization():
    """Test GPU synchronization."""
    print("\n" + "="*70)
    print("TEST 8: GPU Synchronization")
    print("="*70)
    
    if not nv.cuda_is_available():
        print("  No GPU - skipping synchronization test")
        print(" Synchronization test skipped")
        return
    
    nv.set_device('cuda')
    
    # perform operations
    a = nv.array([1, 2, 3, 4, 5])
    b = nv.array([5, 4, 3, 2, 1])
    
    backend = nv.get_backend()
    c = backend.add(a, b)
    
    # synchronize
    nv.synchronize()
    print(" Synchronization successful")
    
    print(" GPU synchronization test passed")


def test_mixed_operations():
    """Test operations with mixed CPU/GPU data."""
    print("\n" + "="*70)
    print("TEST 9: Mixed Operations")
    print("="*70)
    
    if not nv.cuda_is_available():
        print("  No GPU - skipping mixed operations test")
        print(" Mixed operations test skipped")
        return
    
    # create CPU array
    nv.set_device('cpu')
    cpu_data = nv.array([[1, 2, 3], [4, 5, 6]])
    print(f"CPU data shape: {cpu_data.shape}")
    
    # switch to GPU
    nv.set_device('cuda')
    gpu_data = nv.to_device(cpu_data)
    
    # perform operation on GPU
    backend = nv.get_backend()
    result_gpu = backend.sum(gpu_data, axis=0)
    print(f"GPU result: {result_gpu}")
    
    # transfer back to CPU for verification
    result_cpu = nv.to_device(result_gpu, device='cpu')
    expected = np.array([5, 7, 9])
    
    assert np.allclose(result_cpu, expected), "Operation result incorrect"
    print(" Mixed CPU/GPU operations work correctly")
    
    print(" Mixed operations test passed")


def test_performance_comparison():
    """Compare CPU vs GPU performance."""
    print("\n" + "="*70)
    print("TEST 10: Performance Comparison")
    print("="*70)
    
    if not nv.cuda_is_available():
        print("  No GPU - skipping performance comparison")
        print(" Performance comparison skipped")
        return
    
    import time
    
    # test data
    size = 2000
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)
    
    # cPU performance
    nv.set_device('cpu')
    a_cpu = nv.to_device(a)
    b_cpu = nv.to_device(b)
    
    start = time.time()
    backend = nv.get_backend()
    c_cpu = backend.dot(a_cpu, b_cpu)
    cpu_time = (time.time() - start) * 1000
    print(f"CPU time: {cpu_time:.2f} ms")
    
    # gPU performance
    nv.set_device('cuda')
    a_gpu = nv.to_device(a)
    b_gpu = nv.to_device(b)
    
    start = time.time()
    backend = nv.get_backend()
    c_gpu = backend.dot(a_gpu, b_gpu)
    nv.synchronize()
    gpu_time = (time.time() - start) * 1000
    print(f"GPU time: {gpu_time:.2f} ms")
    
    # calculate speedup
    speedup = cpu_time / gpu_time
    print(f"Speedup: {speedup:.2f}x")
    
    # gPU should be faster (or at least not significantly slower)
    # note: For small matrices, CPU might be faster due to overhead
    if size > 1000:
        assert speedup > 1.0, f"GPU not faster for large matrix (speedup: {speedup:.2f}x)"
        print(f" GPU is {speedup:.2f}x faster for large matrices")
    else:
        print(f"  Speedup: {speedup:.2f}x (may vary for small data)")
    
    print(" Performance comparison complete")


def run_all_tests():
    """Run all device management tests."""
    print("\n" + "="*70)
    print(" NEUROVA DEVICE MANAGEMENT TEST SUITE")
    print("="*70)
    
    tests = [
        test_device_detection,
        test_device_switching,
        test_device_context,
        test_array_creation,
        test_array_transfer,
        test_backend_selection,
        test_memory_management,
        test_synchronization,
        test_mixed_operations,
        test_performance_comparison,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f" Test failed: {test_func.__name__}")
            print(f"   Error: {str(e)}")
            failed += 1
    
    # summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n ALL TESTS PASSED! ")
    else:
        print(f"\n  {failed} test(s) failed")
    
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
