#!/usr/bin/env python3
"""
Test P0 fixes locally (no GPU needed for verification)
"""
import sys

def test_backend_parameter():
    """Test that backend parameter exists"""
    print("=" * 60)
    print("TEST 1: Backend Parameter")
    print("=" * 60)
    
    import inspect
    sys.path.insert(0, 'robocache/python')
    import robocache
    
    sig = inspect.signature(robocache.resample_trajectories)
    params = list(sig.parameters.keys())
    
    if 'backend' in params:
        print(f"✅ PASS: backend parameter exists")
        print(f"   Parameters: {params}")
        return True
    else:
        print(f"❌ FAIL: backend parameter missing")
        print(f"   Parameters: {params}")
        return False

def test_cuda_fixture_exists():
    """Test that conftest fixture exists"""
    print("\n" + "=" * 60)
    print("TEST 2: CUDA Fixture")
    print("=" * 60)
    
    try:
        with open('robocache/tests/conftest.py', 'r') as f:
            content = f.read()
        
        if 'def cuda_extension' in content and '@pytest.fixture' in content:
            print("✅ PASS: cuda_extension fixture exists")
            if 'pytest.fail' in content and '_cuda_available' in content:
                print("✅ PASS: Fixture fails if CUDA unavailable")
                return True
            else:
                print("⚠️  WARNING: Fixture may not fail properly")
                return False
        else:
            print("❌ FAIL: cuda_extension fixture missing")
            return False
    except FileNotFoundError:
        print("❌ FAIL: conftest.py not found")
        return False

def test_smoke_test_exists():
    """Test that smoke.py exists with --assert-min-throughput"""
    print("\n" + "=" * 60)
    print("TEST 3: Smoke Test")
    print("=" * 60)
    
    try:
        with open('robocache/benchmarks/smoke.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('--assert-min-throughput' in content, "Has throughput assertion flag"),
            ('backend="cuda"' in content, "Forces CUDA backend"),
            ('torch.cuda.Event' in content, "Uses CUDA events for timing"),
        ]
        
        all_pass = True
        for check, desc in checks:
            if check:
                print(f"✅ PASS: {desc}")
            else:
                print(f"❌ FAIL: {desc}")
                all_pass = False
        
        return all_pass
    except FileNotFoundError:
        print("❌ FAIL: smoke.py not found")
        return False

def test_kernel_inventory():
    """Test that kernel inventory exists"""
    print("\n" + "=" * 60)
    print("TEST 4: Kernel Inventory")
    print("=" * 60)
    
    try:
        with open('robocache/csrc/KERNEL_INVENTORY.md', 'r') as f:
            content = f.read()
        
        if 'csrc/cuda/resample_kernel.cu' in content and 'Canonical' in content:
            print("✅ PASS: KERNEL_INVENTORY.md documents canonical kernels")
            return True
        else:
            print("❌ FAIL: KERNEL_INVENTORY.md incomplete")
            return False
    except FileNotFoundError:
        print("❌ FAIL: KERNEL_INVENTORY.md not found")
        return False

def test_workflows():
    """Test that workflows are fixed"""
    print("\n" + "=" * 60)
    print("TEST 5: Workflows")
    print("=" * 60)
    
    checks = []
    
    # Security scan
    try:
        with open('.github/workflows/security_scan.yml', 'r') as f:
            content = f.read()
        checks.append((
            'pip install -e .' in content,
            "Security scan installs package"
        ))
    except:
        checks.append((False, "Security scan workflow missing"))
    
    # Compute Sanitizer
    try:
        with open('.github/workflows/compute-sanitizer.yml', 'r') as f:
            content = f.read()
        checks.append((
            'compute-sanitizer' in content and 'memcheck' in content,
            "Compute Sanitizer workflow exists"
        ))
    except:
        checks.append((False, "Compute Sanitizer workflow missing"))
    
    # GPU CI
    try:
        with open('.github/workflows/gpu_ci.yml', 'r') as f:
            content = f.read()
        checks.append((
            'smoke.py' in content,
            "GPU CI runs smoke test"
        ))
    except:
        checks.append((False, "GPU CI workflow missing"))
    
    all_pass = True
    for check, desc in checks:
        if check:
            print(f"✅ PASS: {desc}")
        else:
            print(f"❌ FAIL: {desc}")
            all_pass = False
    
    return all_pass

def main():
    print("\n" + "=" * 60)
    print("P0 FIXES VERIFICATION (Local)")
    print("=" * 60 + "\n")
    
    results = [
        test_backend_parameter(),
        test_cuda_fixture_exists(),
        test_smoke_test_exists(),
        test_kernel_inventory(),
        test_workflows(),
    ]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL CHECKS PASSED")
        print("P0 fixes verified locally. Ready for GPU testing.")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
