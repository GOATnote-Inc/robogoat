#!/bin/bash
# Setup GitHub Actions Self-Hosted Runner on Brev GPU Instances
# 
# Usage:
#   H100: brev shell awesome-gpu-name --dir /workspace
#   A100: brev shell a100-gpu-name --dir /workspace
#   Then: bash setup_github_runner.sh <GITHUB_TOKEN> <RUNNER_LABELS>

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <GITHUB_TOKEN> <RUNNER_LABELS>"
    echo "Example: $0 ghp_xxxxx h100,gpu"
    echo ""
    echo "Get token from: https://github.com/GOATnote-Inc/robogoat/settings/actions/runners/new"
    exit 1
fi

GITHUB_TOKEN=$1
RUNNER_LABELS=$2
RUNNER_VERSION="2.311.0"  # Latest as of Nov 2025

echo "==================================="
echo "GitHub Actions Runner Setup (Brev)"
echo "==================================="
echo "Labels: $RUNNER_LABELS"
echo "Version: $RUNNER_VERSION"
echo "==================================="

# Create runner directory
cd /workspace
mkdir -p actions-runner && cd actions-runner

# Download runner
echo "📦 Downloading GitHub Actions runner..."
curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# Extract
echo "📂 Extracting..."
tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
rm actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# Configure runner
echo "⚙️  Configuring runner..."
./config.sh \
    --url https://github.com/GOATnote-Inc/robogoat \
    --token ${GITHUB_TOKEN} \
    --name "brev-$(hostname)-gpu" \
    --labels "${RUNNER_LABELS},self-hosted,brev" \
    --work /workspace/_work \
    --replace \
    --unattended

# Install as tmux session (persistent)
echo "🚀 Starting runner in tmux session..."
tmux new-session -d -s github-runner "cd /workspace/actions-runner && ./run.sh"

echo ""
echo "✅ GitHub Actions runner configured!"
echo ""
echo "Runner is running in tmux session 'github-runner'"
echo "  - Attach: tmux attach -t github-runner"
echo "  - Detach: Ctrl+B, then D"
echo "  - Check: tmux ls"
echo ""
echo "To stop runner:"
echo "  tmux kill-session -t github-runner"
echo ""
echo "View at: https://github.com/GOATnote-Inc/robogoat/settings/actions/runners"

