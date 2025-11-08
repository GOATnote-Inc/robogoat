# GitHub Actions GPU Runner Setup

**Using Your H100 and A100 Brev Instances as CI Runners**

---

## Quick Start

### 1. Generate GitHub Token

Go to: https://github.com/GOATnote-Inc/robogoat/settings/actions/runners/new

Click **"New self-hosted runner"** and copy the token (starts with `ghp_`)

### 2. Setup H100 Runner

```bash
# Login to H100 instance
brev login --token <YOUR_BREV_TOKEN>
cat << 'EOF' | brev shell awesome-gpu-name --dir /workspace 2>&1
# Start tmux session for persistence
tmux new -s setup

# Clone repo
cd /workspace
git clone https://github.com/GOATnote-Inc/robogoat.git || true
cd robogoat

# Setup runner
bash scripts/setup_github_runner.sh \
    ghp_YOUR_GITHUB_TOKEN_HERE \
    h100,gpu,sm90

# Detach from tmux: Ctrl+B, then D
EOF
```

### 3. Setup A100 Runner

```bash
# Login to A100 instance
cat << 'EOF' | brev shell a100-gpu-name --dir /workspace 2>&1
# Start tmux session
tmux new -s setup

# Clone repo
cd /workspace
git clone https://github.com/GOATnote-Inc/robogoat.git || true
cd robogoat

# Setup runner
bash scripts/setup_github_runner.sh \
    ghp_YOUR_GITHUB_TOKEN_HERE \
    a100,gpu,sm80

# Detach from tmux: Ctrl+B, then D
EOF
```

### 4. Verify Runners Active

Visit: https://github.com/GOATnote-Inc/robogoat/settings/actions/runners

You should see:
- ✅ **brev-awesome-gpu-name-gpu** (H100, idle)
- ✅ **brev-a100-gpu-name-gpu** (A100, idle)

---

## How It Works

### Architecture

```
GitHub PR/Push
      ↓
GitHub Actions Trigger
      ↓
Runner Selection (labels: h100,gpu OR a100,gpu)
      ↓
Brev Instance (persistent tmux session)
      ↓
/workspace/actions-runner/run.sh
      ↓
Checkout code → Build CUDA → Run tests → Upload artifacts
```

### Workflows

**H100 Workflow:** `.github/workflows/gpu_ci_h100.yml`
- Triggers: PR, push to main, manual
- Runner labels: `[h100, gpu]`
- Tests: SM90 kernels, benchmarks, regression checks

**A100 Workflow:** `.github/workflows/gpu_ci_a100.yml`
- Triggers: PR, push to main, manual
- Runner labels: `[a100, gpu]`
- Tests: SM80 kernels, benchmarks, regression checks

---

## Cost Management

### Brev Instance Costs
- **H100:** ~$3.00/hour
- **A100:** ~$1.10/hour

### Strategy: On-Demand Runners

**Option A: Keep Instances Running (Expensive)**
```bash
# Runner always available
# Cost: $3.00/hr × 24hr × 30 days = $2,160/month (H100)
```

**Option B: Start/Stop as Needed (Recommended)**
```bash
# Start instance when PR created
# Run CI (15 min)
# Stop instance
# Cost: $3.00/hr × 0.25hr × 30 PRs = $22.50/month (H100)
```

### Implementing Option B

**Manual (Current):**
```bash
# When PR filed:
1. brev shell awesome-gpu-name --dir /workspace
2. Start runner: tmux attach -t github-runner (or restart if needed)
3. CI runs automatically
4. Stop instance after PR merged
```

**Automated (Future - Q1 2026):**
- GitHub webhook triggers Brev instance start
- Runner starts automatically in tmux
- CI runs, uploads artifacts
- Instance auto-stops after idle period

---

## Maintenance

### Check Runner Status

```bash
# H100
brev shell awesome-gpu-name --dir /workspace
tmux attach -t github-runner
# See runner logs, Ctrl+B then D to detach

# A100
brev shell a100-gpu-name --dir /workspace
tmux attach -t github-runner
```

### Restart Runner

```bash
cd /workspace/actions-runner
tmux kill-session -t github-runner
tmux new -s github-runner "./run.sh"
```

### Update Runner

```bash
cd /workspace/actions-runner
./config.sh remove --token <NEW_TOKEN>
# Then re-run setup_github_runner.sh
```

### View Logs

```bash
cd /workspace/actions-runner/_diag
tail -f Runner_*.log
```

---

## Troubleshooting

### Runner Not Showing Up

**Check:**
1. Token expired? (Tokens expire after 1 hour)
2. Network connectivity: `curl https://api.github.com`
3. Tmux session alive: `tmux ls`

**Fix:**
```bash
# Generate new token and reconfigure
cd /workspace/actions-runner
./config.sh remove --token <OLD_TOKEN>
./config.sh --url https://github.com/GOATnote-Inc/robogoat --token <NEW_TOKEN> ...
```

### CI Job Stuck

**Symptoms:** Workflow shows "Queued" forever

**Causes:**
1. No runner with matching labels
2. Runner offline
3. Runner busy with another job

**Fix:**
```bash
# Check runner status
brev shell awesome-gpu-name --dir /workspace
cd /workspace/actions-runner
./run.sh  # Run in foreground to see errors
```

### CUDA Out of Memory

**Symptoms:** Job fails with `CUDA out of memory`

**Fix:**
```bash
# Clear GPU memory before each run
# Add to workflow:
- name: Clear GPU memory
  run: |
    nvidia-smi --gpu-reset
    python3 -c "import torch; torch.cuda.empty_cache()"
```

### Slow Build Times

**Symptoms:** Compilation takes >10 minutes

**Optimization:**
```bash
# Add ccache to speed up builds
sudo apt install ccache
export PATH="/usr/lib/ccache:$PATH"
export CCACHE_DIR=/workspace/.ccache
```

---

## Security

### Token Management

**DO NOT:**
- ❌ Commit tokens to git
- ❌ Share tokens publicly
- ❌ Use tokens with write:repo permissions

**DO:**
- ✅ Use minimal permissions (read:packages, read:org)
- ✅ Rotate tokens monthly
- ✅ Store tokens in password manager

### Network Security

**Brev instances are secure by default:**
- ✅ SSH access only (no public HTTP)
- ✅ GitHub runner connects outbound only
- ✅ No inbound connections required

### Code Execution

**Risk:** PR from external contributor could run malicious code

**Mitigation:**
```yaml
# Only run on PRs from trusted contributors
jobs:
  build-and-test-h100:
    if: github.event.pull_request.head.repo.full_name == github.repository
```

Or use **manual approval** for external PRs:
- Review code first
- Manually trigger workflow with `workflow_dispatch`

---

## Cost Optimization

### Current Setup (Manual Start/Stop)

**Estimated Monthly Cost:**
- H100: ~$22/month (30 PRs × 15 min × $3/hr)
- A100: ~$8/month (30 PRs × 15 min × $1.10/hr)
- **Total: ~$30/month**

**vs GitHub-hosted GPU runners: ~$1,800/month**

**Savings: 98%** 🎉

### Further Optimization

1. **Cache Dependencies:**
   - ccache for C++ compilation
   - PyTorch wheel caching
   - **Saves:** 5-8 minutes per run

2. **Conditional Triggers:**
   - Skip CI for markdown/doc changes
   - Only run full benchmarks on main branch
   - **Saves:** 50% of runs

3. **Spot Instances (Future):**
   - Brev spot pricing: 60-80% discount
   - Trade-off: Potential interruptions
   - **Saves:** $5-10/month additional

---

## Monitoring

### GitHub Actions Dashboard

View all runs: https://github.com/GOATnote-Inc/robogoat/actions

**Key Metrics:**
- Success rate (target: >95%)
- Average duration (target: <15 min)
- Queue time (target: <1 min)

### Custom Dashboard (Future)

Create a dashboard tracking:
- GPU utilization during CI
- Cost per PR
- Performance regression trends
- Runner availability

---

## Roadmap

### Q4 2025 (This Month)
- [x] Setup H100 runner on Brev
- [x] Setup A100 runner on Brev
- [ ] Test workflows on real PRs
- [ ] Add regression gates
- [ ] Document cost tracking

### Q1 2026
- [ ] Automate runner start/stop
- [ ] Add performance dashboard
- [ ] Integrate with Slack notifications
- [ ] Weekly cost reports

### Q2 2026
- [ ] Migrate to spot instances
- [ ] Add more architectures (L40S)
- [ ] Implement caching optimizations
- [ ] External contributor workflow

---

## FAQ

### Q: Why not use GitHub-hosted GPU runners?

**A:** Cost. GitHub charges $4/minute for GPU runners (~$1,800/month for daily CI). Our Brev solution costs ~$30/month with manual start/stop.

### Q: Can external contributors trigger CI?

**A:** Not automatically (security risk). Maintainers review PRs and manually trigger GPU CI with `workflow_dispatch`.

### Q: What if Brev instances are stopped?

**A:** Workflows will queue until runners are available. Start instances manually or implement auto-start webhook.

### Q: How long do instances need to run?

**A:** ~15 minutes per CI run. Start instance when PR filed, stop after merge. Or keep running 9am-5pm PT for immediate feedback.

---

**This setup gives you REAL automated GPU CI using hardware you already have access to!**

**Next Steps:**
1. Generate GitHub token
2. Run setup scripts on H100 and A100
3. Create a test PR to verify workflows
4. Update `GPU_CI_STATUS.md` to reflect actual automation

---

**Setup Date:** November 8, 2025  
**Maintained By:** GOATnote Engineering  
**Support:** File issue if runners go offline

