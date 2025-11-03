# Kubernetes Deployment for RoboCache

This directory contains Kubernetes manifests and Helm charts for deploying RoboCache as a large-scale batch preprocessing pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Kubernetes Cluster                     │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  GPU Node 1  │  │  GPU Node 2  │  │  GPU Node N  │  │
│  │              │  │              │  │              │  │
│  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │  │
│  │  │ Worker │  │  │  │ Worker │  │  │  │ Worker │  │  │
│  │  │  Pod   │  │  │  │  Pod   │  │  │  │  Pod   │  │  │
│  │  │ (H100) │  │  │  │ (H100) │  │  │  │ (H100) │  │  │
│  │  └───┬────┘  │  │  └───┬────┘  │  │  └───┬────┘  │  │
│  └──────┼───────┘  └──────┼───────┘  └──────┼───────┘  │
│         │                 │                 │           │
│         └─────────────────┼─────────────────┘           │
│                           │                             │
│                    ┌──────▼───────┐                     │
│                    │   Shared     │                     │
│                    │   Storage    │                     │
│                    │  (NFS/S3)    │                     │
│                    └──────────────┘                     │
│                                                          │
│                    ┌──────────────┐                     │
│                    │  Prometheus  │                     │
│                    │  Pushgateway │                     │
│                    └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. NVIDIA GPU Operator

Install the NVIDIA GPU Operator to enable GPU support in Kubernetes:

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator
helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator
```

Verify GPU nodes are available:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUs:.status.allocatable."nvidia\.com/gpu"
```

### 2. Persistent Storage

Create a StorageClass for shared dataset storage:

**Option A: NFS Storage**

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs-storage
provisioner: nfs.csi.k8s.io
parameters:
  server: nfs-server.example.com
  share: /data/robot-datasets
mountOptions:
  - hard
  - nfsvers=4.1
```

**Option B: AWS EFS**

```bash
# Install EFS CSI driver
kubectl apply -k "github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=release-1.7"

# Create StorageClass
cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-1234567890abcdef0
  directoryPerms: "700"
EOF
```

### 3. Build and Push Docker Image

```bash
# Build RoboCache container
cd robocache
docker build -t your-registry.com/robocache:latest -f docker/Dockerfile .

# Push to container registry
docker push your-registry.com/robocache:latest
```

## Deployment Methods

### Method 1: Direct Kubernetes Manifests

Deploy the preprocessing job directly:

```bash
# Create namespace
kubectl create namespace robot-learning

# Create S3 credentials secret (if using S3)
kubectl create secret generic s3-credentials \
  --from-literal=access-key-id=YOUR_ACCESS_KEY \
  --from-literal=secret-access-key=YOUR_SECRET_KEY \
  -n robot-learning

# Deploy the job
kubectl apply -f preprocessing_job.yaml

# Monitor progress
kubectl get jobs -n robot-learning -w
kubectl logs -f job/robocache-preprocess-rtx -n robot-learning
```

### Method 2: Helm Chart (Recommended)

Deploy using Helm for easier configuration management:

```bash
# Install the chart
helm install robocache ./helm-chart \
  --namespace robot-learning \
  --create-namespace \
  --set image.repository=your-registry.com/robocache \
  --set preprocessing.inputDir=/data/input/rtx \
  --set preprocessing.outputDir=/data/output/rtx-50hz \
  --set job.parallelism=8 \
  --set gpu.type=h100

# Check deployment status
helm status robocache -n robot-learning

# View logs
kubectl logs -f -l app=robocache -n robot-learning
```

## Configuration

### Scaling

Adjust parallelism based on available GPU nodes:

```bash
# For 8 GPU nodes
helm upgrade robocache ./helm-chart \
  --set job.parallelism=8 \
  --reuse-values

# For 16 GPU nodes
helm upgrade robocache ./helm-chart \
  --set job.parallelism=16 \
  --reuse-values
```

### Resource Allocation

Customize CPU, memory, and GPU resources:

```bash
helm upgrade robocache ./helm-chart \
  --set resources.requests.cpu=16 \
  --set resources.requests.memory=128Gi \
  --set gpu.count=2 \
  --reuse-values
```

### Storage

Configure dataset storage:

```bash
helm upgrade robocache ./helm-chart \
  --set storage.size=20Ti \
  --set storage.storageClass=efs-sc \
  --reuse-values
```

## Monitoring

### View Job Progress

```bash
# Job status
kubectl get jobs -n robot-learning

# Pod status
kubectl get pods -n robot-learning -l app=robocache

# Detailed job info
kubectl describe job robocache-preprocess-rtx -n robot-learning
```

### Access Logs

```bash
# Stream logs from all pods
kubectl logs -f -l app=robocache -n robot-learning

# Logs from specific pod
kubectl logs -f <pod-name> -n robot-learning

# Previous pod logs (if pod restarted)
kubectl logs --previous <pod-name> -n robot-learning
```

### Prometheus Metrics

If observability is enabled, metrics are pushed to Prometheus Pushgateway:

```bash
# Port-forward to Pushgateway
kubectl port-forward svc/prometheus-pushgateway 9091:9091 -n robot-learning

# View metrics
curl http://localhost:9091/metrics | grep robocache
```

Key metrics:
- `robocache_trajectories_processed_total`: Total trajectories processed
- `robocache_resample_duration_seconds`: Resampling latency histogram
- `robocache_bandwidth_gbps`: GPU memory bandwidth utilization

## Troubleshooting

### Pods Stuck in Pending

Check GPU availability:

```bash
kubectl describe pod <pod-name> -n robot-learning

# Common issues:
# 1. No GPU nodes available
# 2. Insufficient GPU resources
# 3. Node taints not tolerated
```

### Out of Memory Errors

Increase shared memory size:

```bash
helm upgrade robocache ./helm-chart \
  --set storage.shmSize=64Gi \
  --reuse-values
```

Or adjust batch size:

```bash
helm upgrade robocache ./helm-chart \
  --set preprocessing.batchSize=128 \
  --reuse-values
```

### Slow I/O Performance

Check storage performance:

```bash
# Test read speed
kubectl exec -it <pod-name> -n robot-learning -- \
  dd if=/data/input/test-file of=/dev/null bs=1M count=1000

# Test write speed
kubectl exec -it <pod-name> -n robot-learning -- \
  dd if=/dev/zero of=/data/output/test-file bs=1M count=1000
```

Optimize storage:
1. Use local NVMe for temporary data
2. Enable NFS async mount option
3. Increase NFS read/write size
4. Use S3 Transfer Acceleration (if using S3)

### Job Not Completing

Check for failed pods:

```bash
kubectl get pods -n robot-learning -l app=robocache --field-selector=status.phase=Failed

# Debug failed pod
kubectl logs <failed-pod-name> -n robot-learning
kubectl describe pod <failed-pod-name> -n robot-learning
```

## Performance Benchmarks

Expected throughput on DGX H100 (8×H100 GPUs):

| Configuration | Parallelism | Throughput (traj/s) | Time for 1M Traj |
|---------------|-------------|---------------------|------------------|
| 1×H100 (BF16) | 1           | 31,200              | 32 seconds       |
| 8×H100 (BF16) | 8           | 224,000             | 4.5 seconds      |
| 8×H100 (FP32) | 8           | 148,000             | 6.8 seconds      |

Actual throughput depends on:
- Storage I/O speed (NFS, S3, or local storage)
- Network bandwidth (if using distributed storage)
- CPU preprocessing overhead
- Batch size and dataset characteristics

## Best Practices

1. **Use BF16 precision** for H100 GPUs (1.7x faster than FP32)
2. **Optimize batch size** based on GPU memory (256-512 recommended)
3. **Use local storage** for temporary files to reduce network I/O
4. **Monitor GPU utilization** using `nvidia-smi` or DCGM
5. **Set appropriate resource limits** to avoid pod eviction
6. **Use pod anti-affinity** to distribute pods across nodes
7. **Enable monitoring** with Prometheus for production deployments

## Cleanup

```bash
# Uninstall Helm release
helm uninstall robocache -n robot-learning

# Or delete direct deployment
kubectl delete -f preprocessing_job.yaml

# Delete namespace (removes all resources)
kubectl delete namespace robot-learning
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/robocache/issues
- Documentation: https://github.com/yourusername/robocache/docs
