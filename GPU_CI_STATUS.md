# GPU CI/CD Infrastructure

**Last Updated:** November 8, 2025  
**Status:** Self-Hosted GPU Runners (H100 + A100)

---

## Infrastructure

### Self-Hosted GPU Runners

**H100 Runner (SM90)**
- **Workflow:** `.github/workflows/gpu_ci_h100.yml`
- **Hardware:** Brev H100 PCIe instance
- **Labels:** `[h100, gpu, sm90]`
- **Triggers:** PR, push to main, manual dispatch
- **Setup:** `scripts/setup_github_runner.sh`

**A100 Runner (SM80)**
- **Workflow:** `.github/workflows/gpu_ci_a100.yml`
- **Hardware:** Brev A100 SXM4 instance
- **Labels:** `[a100, gpu, sm80]`
- **Triggers:** PR, push to main, manual dispatch
- **Setup:** `scripts/setup_github_runner.sh`

**CPU CI (Linting & Fallbacks)**
- **Workflow:** `.github/workflows/ci.yml`
- **Runs on:** GitHub-hosted `ubuntu-latest`
- **Tests:** Linting, type checking, CPU fallback correctness

### CI Pipeline

**Per PR:**
1. CPU CI runs immediately (5 min)
2. H100 CI runs on self-hosted runner (15 min)
3. A100 CI runs on self-hosted runner (15 min)
4. All must pass before merge

**Artifacts:**
- Benchmark CSVs uploaded to GitHub Actions
- Performance reports generated automatically
- Regression detection with baseline comparison

---

## Cost Management

**Current Infrastructure:**
- Self-hosted runners on existing Brev instances
- On-demand: Start instances for CI, stop after completion
- Estimated: ~$30/month for typical PR volume

**vs Alternatives:**
- GitHub-hosted GPU runners: ~$1,800/month
- Dedicated 24/7 cloud instance: ~$2,000/month

**Optimization:**
- Use tmux for persistent runner sessions
- Start/stop instances as needed
- Cache builds to reduce compilation time

---

## External Contributions

**For PRs from external contributors:**
- CPU CI runs automatically
- GPU CI requires maintainer approval (security)
- Provide benchmark results from your own hardware when possible

**Setup guide:** See `docs/GPU_RUNNER_SETUP.md` for detailed instructions

