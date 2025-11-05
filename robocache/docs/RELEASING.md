# Releasing RoboCache

This document describes the release process for RoboCache.

## Release Checklist

### Pre-Release

- [ ] All tests passing on CI (main branch)
- [ ] Documentation is up-to-date
- [ ] CHANGELOG.md updated with release notes
- [ ] Version bumped in `python/robocache/_version.py`
- [ ] Performance benchmarks re-run and validated
- [ ] NCU profiling data updated (if kernel changes)
- [ ] Security scan passed (Trivy)
- [ ] No known critical bugs

### Release Process

1. **Create Release Branch**
   ```bash
   git checkout -b release/v0.2.1
   ```

2. **Update Version**
   Edit `python/robocache/_version.py`:
   ```python
   __version__ = "0.2.1"
   __build_date__ = "2025-11-05"
   __git_commit__ = "abc123"  # Will be updated by CI
   ```

3. **Update CHANGELOG**
   Add release notes to `CHANGELOG.md`:
   ```markdown
   ## [0.2.1] - 2025-11-05
   
   ### Added
   - Production-grade API with multi-backend support
   - Comprehensive test suite and CI pipeline
   - Wheel building for PyPI distribution
   
   ### Changed
   - Improved DRAM bandwidth from 1.59% to 23.76% on H100
   
   ### Fixed
   - BF16 conversion errors in multimodal fusion
   ```

4. **Commit and Push**
   ```bash
   git add python/robocache/_version.py CHANGELOG.md
   git commit -m "Bump version to 0.2.1"
   git push origin release/v0.2.1
   ```

5. **Create Pull Request**
   - Open PR from `release/v0.2.1` to `main`
   - Wait for CI to pass
   - Get review and approval
   - Merge to main

6. **Create Git Tag**
   ```bash
   git checkout main
   git pull
   git tag -a v0.2.1 -m "Release v0.2.1"
   git push origin v0.2.1
   ```

7. **Create GitHub Release**
   - Go to GitHub Releases page
   - Click "Draft a new release"
   - Select tag `v0.2.1`
   - Title: "RoboCache v0.2.1"
   - Description: Copy from CHANGELOG.md
   - Attach artifacts:
     - Source tarball
     - Pure Python wheel
     - CUDA wheels (if applicable)
     - SBOM (Software Bill of Materials)
   - Publish release

8. **Automated PyPI Upload**
   - GitHub Actions will automatically build and upload wheels
   - Triggered by release creation
   - Uses trusted publishing (no API token needed)

9. **Verify PyPI Upload**
   ```bash
   pip install --upgrade robocache
   python -c "import robocache; robocache.print_installation_info()"
   ```

10. **Announce Release**
    - Update README badges (if changed)
    - Post on community channels
    - Update documentation site

### Post-Release

- [ ] Verify PyPI package works on fresh install
- [ ] Update version to next dev version (e.g., `0.2.2.dev0`)
- [ ] Monitor for bug reports
- [ ] Update dependent projects

## Version Numbering

RoboCache follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (0.x.x): Backward-incompatible API changes
- **MINOR** (x.1.x): New features, backward-compatible
- **PATCH** (x.x.1): Bug fixes, backward-compatible

### API Version

The `__api_version__` tracks API compatibility separately:
- Same API version = backward compatible
- Different API version = potentially breaking changes

Example:
```python
__version__ = "0.2.1"  # Full package version
__api_version__ = "0.2"  # API compatibility version
```

## Release Cadence

- **Patch releases**: As needed for bug fixes (weekly if needed)
- **Minor releases**: Monthly for new features
- **Major releases**: When significant API changes are needed

## Hotfix Process

For critical bugs in released versions:

1. Create hotfix branch from release tag:
   ```bash
   git checkout -b hotfix/v0.2.2 v0.2.1
   ```

2. Fix bug and update version to patch release

3. Follow normal release process

4. Merge back to main and develop

## Rollback Process

If a release has critical issues:

1. **Yank from PyPI** (doesn't delete, but hides from `pip install`):
   ```bash
   pip install twine
   twine yank robocache --version 0.2.1
   ```

2. **Create hotfix release** with fix

3. **Document in CHANGELOG**

## Testing Releases

### Test PyPI

Before releasing to production PyPI, test on TestPyPI:

1. Build wheels:
   ```bash
   cd robocache
   python -m build
   ```

2. Upload to TestPyPI:
   ```bash
   twine upload --repository testpypi dist/*
   ```

3. Install and test:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ robocache
   ```

## Wheel Variants

RoboCache provides multiple wheel variants:

1. **Pure Python** (`robocache-0.2.1-py3-none-any.whl`):
   - PyTorch backend only
   - Works on all platforms (Linux, macOS, Windows)
   - No CUDA required

2. **manylinux CUDA** (`robocache-0.2.1-cp310-cp310-manylinux_2_17_x86_64.whl`):
   - Includes precompiled CUDA kernels
   - Linux only
   - Requires matching CUDA version (11.8 or 12.1)
   - Separate wheels for each Python version

3. **Source Distribution** (`robocache-0.2.1.tar.gz`):
   - For platforms not covered by wheels
   - Builds CUDA kernels at install time
   - Requires CUDA toolkit and C++ compiler

## Troubleshooting

### Build Failures

If wheel build fails:
- Check CUDA version compatibility
- Ensure PyTorch is installed
- Verify CMake and compilers are available

### Upload Failures

If PyPI upload fails:
- Check PyPI API token is valid
- Verify version number is unique (not already uploaded)
- Ensure package metadata is valid (`twine check`)

### Installation Issues

If users report installation problems:
- Check compatibility matrix (Python, CUDA, OS)
- Verify wheel platform tags
- Test in clean virtualenv

## Security

- All releases are signed (GPG)
- SBOM (Software Bill of Materials) provided
- Vulnerability scans performed before release
- Security advisories published via GitHub Security

## Contact

For release questions:
- GitHub Issues: https://github.com/GOATnote-Inc/robogoat/issues
- Email: b@thegoatnote.com

