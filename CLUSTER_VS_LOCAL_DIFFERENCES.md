# Analysis of Cluster vs. Local Processing Performance

This document summarizes the key differences in the processing environment and configuration between the **Euler Cluster** and **Local** execution that explain the observed performance degradation on the cluster.

## 1. Major Bottlenecks (The "Big Three")

### 🔴 Redundant I/O Stage (The "Scratch" Trap)
*   **Local:** Directly reads files from a fast local SSD.
*   **Cluster:** Data is on a networked filesystem.
*   **The Issue:** The code was configured with `move_to_scratch: true`. While intended to speed up access, it actually performs a `shutil.copy` from the network to local scratch before starting.
*   **The "RAM" Factor:** Since you are using `DayRAMDataset` or `H5RAMDataset` on the cluster, the training process **loads the entire data file into RAM once** at the start.
*   **Impact:** Moving to scratch first effectively causes the same data to be read from the network **twice** (once for the copy, once to load into RAM). By disabling `move_to_scratch`, we eliminate the redundant copy step and load directly from the network into RAM in one pass.

### 🔴 Synchronous CUDA Execution
*   **Local:** Runs CUDA kernels asynchronously (default behavior).
*   **Cluster:** The generated SBATCH scripts (`generate_independent_jobs.sh`) explicitly set `export CUDA_LAUNCH_BLOCKING=1`.
*   **The Issue:** This environment variable forces the CPU to wait for every single GPU kernel to finish before proceeding. It is only intended for debugging and can slow down training by **5x-10x**.

### 🔴 CPU Resource Starvation
*   **Local:** Has access to all available CPU cores for data loading and preprocessing.
*   **Cluster:** The SBATCH scripts request `#SBATCH --ntasks=1` but set `export OMP_NUM_THREADS=8`.
*   **The Issue:** Unless `--cpus-per-task=8` is also specified, the SLURM scheduler may only allocate 1 physical core to the task. Having 8 threads fighting for 1 physical core causes massive **context-switching overhead**, making data loading extremely slow.

---

## 2. Configuration & Framework Differences

| Feature | Local Environment | Cluster Environment | Impact on Performance |
| :--- | :--- | :--- | :--- |
| **`CUDA_LAUNCH_BLOCKING`** | 0 (Normal) | **1 (Synchronous)** | **Critical Slowdown.** GPU can't pipeline kernels. |
| **`scratch_dir`** | Local path | `/cluster/work/...` | No I/O gain from "scratch" movement. |
| **`move_to_scratch`** | Usually `true` | `true` | Double network I/O overhead on cluster. |
| **`deterministic`** | `True` | `True` | Both are slowed down by `cudnn.benchmark = False`. |
| **CPU Cores** | All available | **1 (Requested)** vs **8 (Threads)** | Severe CPU bottleneck in data loaders. |
| **Python Mode** | Interactive/Standard | `python -u` (Unbuffered) | Negligible; fine for logging. |
| **Wandb Mode** | `online` | `offline` | Cluster avoids networking lag for logging. |

---

## 3. Recommended Fixes for Euler

To match or exceed local performance on the cluster, the following changes are recommended:

1.  **Fix sbatch templates:**
    *   Remove `export CUDA_LAUNCH_BLOCKING=1`.
    *   Add `#SBATCH --cpus-per-task=8`.
2.  **Fix Cluster Config (`config_cluster.yaml` / `config_cluster_mao_laplacian.yaml`):**
    *   Change `scratch_dir` to point to a local path like `/scratch/` or `$TMPDIR/` instead of `/cluster/work/`.
    *   Alternatively, set `move_to_scratch: false` if the network filesystem is fast enough for direct reads.
3.  **Optimize PyTorch:**
    *   In `src/main.py`, consider setting `torch.backends.cudnn.benchmark = True` for training if reproducibility is secondary to speed.
    *   Increase `num_workers` in the DataLoader to match the allocated `--cpus-per-task`.
