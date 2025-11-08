# GPU CI Security for Public Repository

**Date:** November 8, 2025  
**Status:** Manual approval required for external PRs

---

## Security Model

### The Problem

Self-hosted runners on public repos are a security risk:
- External PRs can execute arbitrary code on your runners
- Malicious code could steal secrets, mine crypto, or attack infrastructure
- GitHub warns against this: https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners#self-hosted-runner-security

### Our Solution

**Three-Tier Security:**

1. **Workflow Dispatch Only (Default)**
   - GPU CI only runs when manually triggered by maintainers
   - No automatic execution on PRs/pushes
   
2. **Same-Repo PRs Only**
   - If enabled, only PRs from branches in `GOATnote-Inc/robogoat` run
   - Forks are blocked by `if: github.event.pull_request.head.repo.full_name == github.repository`
   
3. **Environment Protection**
   - `environment: gpu-runners` requires maintainer approval
   - Even same-repo PRs wait for manual review

---

## Workflow Configuration

### Current Setup (Secure)

```yaml
on:
  workflow_dispatch:  # Manual only

jobs:
  build-and-test-h100:
    runs-on: self-hosted
    if: |
      github.event_name == 'workflow_dispatch' ||
      (github.event.pull_request.head.repo.full_name == github.repository)
    environment:
      name: gpu-runners
```

**What this does:**
- ✅ Maintainers manually trigger GPU CI via Actions tab
- ✅ PRs from forks are blocked automatically
- ✅ Environment protection adds approval gate
- ❌ No automatic execution on every PR (by design)

---

## External Contributor Workflow

### For External PRs (From Forks)

**Current Process:**
1. External contributor opens PR from fork
2. CPU CI runs automatically (safe, GitHub-hosted)
3. GPU CI does NOT run (blocked)
4. Maintainer reviews code
5. If safe, maintainer manually triggers GPU CI:
   - Go to Actions tab
   - Select "GPU CI - H100" or "GPU CI - A100"
   - Click "Run workflow"
   - Select the PR branch
   - Click "Run workflow"

**Why manual?**
- We control what code runs on our expensive GPU hardware
- Prevents malicious PRs from stealing compute
- Standard practice for public repos with self-hosted runners

### For Internal PRs (Same Repo)

**If you trust your team:**

Uncomment these lines in workflows:
```yaml
on:
  workflow_dispatch:
  pull_request:        # Enable this
    branches: [main]
  push:                # Enable this
    branches: [main]
```

Then GPU CI will run automatically on:
- ✅ PRs from branches in `GOATnote-Inc/robogoat`
- ❌ PRs from forks (still blocked)

---

## Setup Environment Protection (Recommended)

### Step 1: Create Environment

1. Go to: https://github.com/GOATnote-Inc/robogoat/settings/environments
2. Click "New environment"
3. Name: `gpu-runners`
4. Add protection rules:
   - ✅ Required reviewers: Add yourself and team
   - ✅ Wait timer: 0 minutes (optional)
   - ✅ Deployment branches: `main` only

### Step 2: Configure Workflow

Already done - workflows use `environment: gpu-runners`

### Step 3: Test

1. Create a test PR
2. GPU CI will show "Waiting for approval"
3. Maintainer reviews and approves
4. GPU CI runs

---

## Alternative: Organization Runners

**For larger teams:**

GitHub Enterprise allows organization-level runners with:
- Runner groups (isolate public/private repos)
- Granular access controls
- Audit logging

**Cost:** $21/user/month (GitHub Enterprise)

**Our choice:** Manual approval (free, secure enough for small team)

---

## Monitoring

### Check Runner Activity

```bash
# SSH to Brev instance
brev shell awesome-gpu-name --dir /workspace

# View runner logs
cd /workspace/actions-runner/_diag
tail -f Runner_*.log
```

### Audit Workflow Runs

- https://github.com/GOATnote-Inc/robogoat/actions
- Filter by workflow: "GPU CI - H100" or "GPU CI - A100"
- Check "Triggered by" - should only be maintainers

---

## Incident Response

### If Malicious Code Runs

1. **Immediately:**
   - Kill runner: `tmux kill-session -t github-runner`
   - Stop Brev instance
   - Revoke GitHub runner token

2. **Investigate:**
   - Check `_diag/Runner_*.log` for executed commands
   - Review git history of suspicious PR
   - Check for crypto miners: `top`, `nvidia-smi`

3. **Recover:**
   - Destroy Brev instance
   - Create fresh instance
   - Re-setup runner with new token
   - Update repository secrets

4. **Prevent:**
   - Ban malicious contributor
   - Add stricter branch protection
   - Review all PRs more carefully

---

## Best Practices

### ✅ DO

- Review all external PRs before triggering GPU CI
- Use manual `workflow_dispatch` for untrusted code
- Monitor runner logs regularly
- Keep runners updated
- Use dedicated instances (don't share with production)

### ❌ DON'T

- Enable automatic PR triggers for forks
- Share runner tokens publicly
- Run runners as root
- Allow runners to access production secrets
- Use runners with write permissions to repo

---

## Comparison to Other Approaches

### Option 1: GitHub-Hosted GPU Runners

**Pros:**
- Fully isolated per job
- No security concerns
- Automatic scaling

**Cons:**
- $4/minute (~$1,800/month)
- Limited GPU options

**Verdict:** Too expensive for our volume

---

### Option 2: Ephemeral Self-Hosted Runners

**Concept:** Spin up fresh instance per job, destroy after

**Pros:**
- Better isolation than persistent runners
- Harder to compromise

**Cons:**
- Startup overhead (5-10 min per job)
- Complex automation
- Still need runner security

**Verdict:** Future improvement (Q2 2026)

---

### Option 3: Manual Validation Only

**Current approach before GPU CI:**

**Pros:**
- Perfect security (manual review)
- Full control

**Cons:**
- Slow (days to validate)
- Doesn't scale

**Verdict:** GPU CI with manual approval is better

---

## FAQ

### Q: Can I make GPU CI fully automatic?

**A:** Not safely on a public repo. You can enable it for same-repo PRs, but forks must remain manual.

### Q: What if I make the repo private?

**A:** Then you can safely enable automatic triggers. GitHub restricts forks on private repos.

### Q: Do other projects do this?

**A:** Yes. PyTorch, TensorFlow, and other projects with GPU CI use manual approval or private repos.

### Q: Can't I just sandbox the runner?

**A:** Docker/VMs help, but sophisticated attacks can escape. Manual approval is safest for public repos.

---

## Summary

**Current security model:**
- ✅ Manual approval for all GPU CI (safe)
- ✅ Blocks all fork PRs automatically
- ✅ Maintainers control when runners execute
- ✅ No risk of malicious code on GPU hardware

**Trade-off:**
- ⏱️ Slightly slower (manual trigger)
- 👤 Requires maintainer action

**Benefit:**
- 💰 98% cheaper than GitHub-hosted ($30 vs $1,800/month)
- 🔒 Full security control
- 🎛️ Flexibility to approve/reject

**This is the industry-standard approach for public repos with self-hosted runners.**

---

**References:**
- https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners#self-hosted-runner-security
- https://github.blog/changelog/2021-04-22-github-actions-self-hosted-runners-can-now-disable-automatic-updates/
- https://www.theregister.com/2024/02/20/github_actions_worm/

**Last Updated:** November 8, 2025

