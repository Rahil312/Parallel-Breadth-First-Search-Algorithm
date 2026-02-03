# System Requirements

## Hardware Requirements

### CPU (OpenMP)
- **Minimum**: 2-core x86_64 processor
- **Recommended**: 8+ cores with hyperthreading
- **Memory**: 4GB RAM minimum, 16GB+ recommended

### Cluster (MPI) 
- **Nodes**: 2+ compute nodes
- **Network**: Low-latency interconnect (InfiniBand preferred)
- **Memory**: 8GB+ per node recommended

### GPU (CUDA)
- **Minimum**: NVIDIA GPU with Compute Capability 6.0+
- **Recommended**: Tesla V100, A100, or RTX 30/40 series
- **Memory**: 8GB+ GPU memory for large graphs

## Software Dependencies

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    gcc g++ \
    libopenmpi-dev openmpi-bin \
    make \
    git
```

### CentOS/RHEL/Fedora
```bash
# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install openmpi openmpi-devel
module load mpi

# Fedora
sudo dnf groupinstall "Development Tools"
sudo dnf install openmpi openmpi-devel
```

### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install dependencies via Homebrew
brew install gcc open-mpi make
```

### CUDA Installation
```bash
# Download from: https://developer.nvidia.com/cuda-downloads
# Or via package manager:

# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda
```

## Environment Setup

### MPI Environment
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH=$PATH:/usr/lib64/openmpi/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/openmpi/lib

# Verify installation
mpirun --version
```

### CUDA Environment
```bash
# Add to ~/.bashrc or ~/.zshrc  
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version
nvidia-smi
```

### OpenMP Environment
```bash
# Set thread count (optional, defaults to all available cores)
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

## Performance Tuning

### System-level Optimizations
```bash
# Disable CPU frequency scaling for consistent benchmarks
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set CPU affinity for MPI processes
mpirun --bind-to core --map-by core ./mpi_bfs

# NUMA awareness
numactl --interleave=all ./omp_bfs
```

### Compiler Optimizations
```bash
# Build with architecture-specific optimizations
make CXXFLAGS="-O3 -march=native -mtune=native"

# Enable link-time optimization
make CXXFLAGS="-O3 -flto -march=native"
```