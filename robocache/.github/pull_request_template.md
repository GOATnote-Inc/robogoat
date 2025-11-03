# Pull Request

## Description

<!-- Provide a brief description of your changes -->

### Type of Change

Please check the relevant options:

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] ⚡ Performance improvement (optimization that improves throughput/latency)
- [ ] 🔨 Refactoring (code restructuring without changing functionality)
- [ ] 📚 Documentation update
- [ ] 🧪 Test addition or improvement
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)

### Related Issues

<!-- Link related issues here -->
Closes #(issue number)
Related to #(issue number)

## Changes Made

<!-- Provide a detailed list of changes -->

- Change 1
- Change 2
- Change 3

## Motivation and Context

<!-- Why is this change required? What problem does it solve? -->

## Performance Impact

<!-- For performance-related changes, provide benchmark results -->

**Benchmarks** (if applicable):

```
Configuration: batch=256, source_len=100, target_len=50, action_dim=32
GPU: NVIDIA H100

Before:
- Throughput: X K samples/sec
- Bandwidth: Y GB/s

After:
- Throughput: X K samples/sec (Z% improvement)
- Bandwidth: Y GB/s (Z% improvement)
```

**Memory Usage** (if applicable):
- Peak memory: X GB
- Memory increase/decrease: Y GB

## Testing

### Test Environment

- [ ] Tested on H100
- [ ] Tested on A100
- [ ] Tested on RTX 4090
- [ ] Other GPU: ____________

### Test Results

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Benchmarks run successfully
- [ ] Examples work as expected

**Test Commands Run:**

```bash
# List the test commands you ran
pytest tests/ -v
cd build && ./benchmark_trajectory_resample
python examples/basic_usage.py
```

**Test Output:**

```
<!-- Paste relevant test output here -->
```

## Code Quality

- [ ] Code follows the style guidelines ([CONTRIBUTING.md](../CONTRIBUTING.md))
- [ ] Self-review of code performed
- [ ] Code has been commented, particularly in complex areas
- [ ] Documentation has been updated (README, docstrings, etc.)
- [ ] No new warnings generated
- [ ] Pre-commit hooks pass (black, flake8, mypy)

## Documentation

- [ ] README.md updated (if needed)
- [ ] API documentation updated (docstrings)
- [ ] Examples updated or added
- [ ] Technical documentation updated (docs/ folder)
- [ ] CHANGELOG.md updated (for version releases)

## Breaking Changes

<!-- If this PR introduces breaking changes, list them here -->

- [ ] This PR introduces breaking changes

**Breaking Changes Details:**
<!-- Describe what breaks and how users should migrate -->

## Additional Notes

<!-- Any additional information that reviewers should know -->

## Screenshots/Visualizations (if applicable)

<!-- Add screenshots, plots, or visualizations if relevant -->

## Checklist

- [ ] I have read the [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings or errors
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
- [ ] I have checked my code and corrected any misspellings

## Reviewer Notes

<!-- Anything specific you want reviewers to focus on? -->

**Areas needing special attention:**
-

**Questions for reviewers:**
-

---

**Thank you for contributing to RoboCache!** 🚀
