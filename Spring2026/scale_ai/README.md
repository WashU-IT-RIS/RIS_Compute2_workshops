# Scaling AI with GPU Profiling on RIS Compute2

This repository provides scripts, utilities, and sample code for profiling GPU workloads on the Compute2 cluster. It is designed to help users benchmark, analyze, and optimize GPU-accelerated machine learning and deep learning applications using tools such as NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`).

## Repository Structure

- `src/` — Source code and scripts for profiling and benchmarking.
  - `*.py` — Python scripts for running and profiling ML workloads (e.g., `train.py`, `train_scale.py`).
  - `*.sh` — SLURM batch scripts and shell utilities for submitting jobs to Compute2 
- `test/` — (Reserved for test scripts or test data.)
- `LICENSE` — License information for the repository.

## Main Features

- **Profiling Scripts:** Ready-to-use SLURM scripts for running and profiling PyTorch on NVIDIA GPUs.
- **Sample Workloads:** Example scripts for deep learning (ResNet50 training) and RAPIDS cuML t-SNE dimensionality reduction.
- **NVTX Annotations:** Key code sections are annotated for detailed profiling with NVIDIA tools.

## Getting Started

1. Clone the repository to your home or scratch directory.
2. Review and edit the SLURM scripts in `src/` to match your environment or profiling needs.
3. Submit jobs using `sbatch` (e.g., `sbatch src/<job>.sh`).
4. Analyze the output files and profiling results in the generated logs and output files.

## Requirements

- Access to the Compute2 (or similar SLURM-based GPU cluster) with support of Apptainer.
- NVIDIA GPU (A100, H100 or any other Comptaible)
- Modules: CUDA, PyTorch, RAPIDS cuML, Nsight Systems, Nsight Compute etc. (see SLURM scripts for specific versions)

## Authorship

- **Primary Author:** Ayush Chaturvedi

## License

This repository is licensed under the terms of the LICENSE file provided.


## Contact

For questions or suggestions, please open an issue or contact the repository author.

## Dataset Download: 

Tiny-ImageNet: 
https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet?resource=download 

Use the curl command: 

```
#!/bin/bash
curl -L -o tiny-imagenet.zip\
  https://www.kaggle.com/api/v1/datasets/download/akash2sharma/tiny-imagenet 
```

> Recommended to use $SCRACTH directory
