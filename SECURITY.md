# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@thegoatnote.com**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

## Security Measures

### Build & Distribution
- **SBOM Generation:** Every release includes a CycloneDX Software Bill of Materials
- **Dependency Scanning:** Weekly automated scans with `pip-audit` and Trivy
- **Signed Artifacts:** PyPI wheels are signed with Sigstore (planned)
- **SLSA Level 3:** Build provenance attestation (partial, in progress)

### CUDA Kernels
- **Memory Safety:** All kernels use bounds checking on array accesses
- **Compute Sanitizer:** CI runs racecheck and memcheck on all CUDA code
- **Input Validation:** All public APIs validate tensor shapes, dtypes, and devices

### Data Handling
- **No Data Retention:** RoboCache processes data in-memory only
- **No Telemetry:** No usage data or metrics are sent externally
- **Local Execution:** All operations run locally on user hardware

### Dependencies
- **Minimal Surface:** Core dependencies limited to PyTorch and NumPy
- **Pinned Versions:** All dependencies pinned in `pyproject.toml`
- **Automated Updates:** Dependabot monitors for security updates

## Vulnerability Disclosure Timeline

1. **T+0:** Vulnerability reported to security@thegoatnote.com
2. **T+48h:** Acknowledgment sent to reporter
3. **T+7d:** Initial assessment and severity classification
4. **T+30d:** Fix developed and tested
5. **T+45d:** Coordinated disclosure with reporter
6. **T+60d:** Public disclosure and patched release

## Known Limitations

### CUDA-Specific
- **GPU OOM:** Large inputs can cause GPU out-of-memory errors; users should validate input sizes
- **Driver Compatibility:** Requires NVIDIA driver ≥ 520.xx for CUDA 12.x/13.x
- **FP16/BF16 Precision:** Reduced precision may impact numerical stability in edge cases

### ROS 2 Integration
- **Network Security:** ROS 2 DDS default configuration is not secure; use SROS2 for production
- **Message Validation:** Input sensor messages are not cryptographically verified

## Security Best Practices

### For Developers
```bash
# Always use latest security patches
pip install --upgrade robocache

# Run security scans locally
pip-audit
trivy fs .

# Enable Compute Sanitizer during development
compute-sanitizer --tool memcheck python your_script.py
```

### For Production Deployments
```python
# Validate inputs before processing
def validate_pointcloud(points: torch.Tensor) -> bool:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Invalid point cloud shape")
    if points.shape[0] > 10_000_000:  # 10M point limit
        raise ValueError("Point cloud too large")
    if not torch.isfinite(points).all():
        raise ValueError("Point cloud contains NaN/Inf")
    return True

# Use CPU fallbacks for untrusted inputs
robocache.voxelize_pointcloud(points, device='cpu')
```

### For ROS 2 Integration
```bash
# Enable SROS2 security
ros2 security create_keystore demo_keys
ros2 security create_key demo_keys /robocache_preprocessor

# Run with security enabled
ros2 run robocache_ros robot_preprocessor.py \
  --ros-args --enclave /robocache_preprocessor
```

## Acknowledgments

We thank the security research community for responsible disclosure of vulnerabilities.

### Hall of Fame
*No vulnerabilities reported yet*

## Contact

- **Email:** security@thegoatnote.com
- **PGP Key:** Available upon request
- **Response Time:** < 48 hours

---

**Last Updated:** November 7, 2025
