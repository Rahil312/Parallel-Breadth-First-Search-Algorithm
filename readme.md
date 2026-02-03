## Parallel BFS

---

### 1 . Project Purpose  

This repository implements **Breadth-First Search (BFS)** on three different parallel-computing back-ends so you can compare their speed, memory behaviour, and scalability on the same graph:

| Backend | Model | File | Highlights |
|---------|-------|------|------------|
| **OpenMP** | shared memory, x86-64 | `openmp.cpp` | top-down **and** direction-optimising BFS with dynamic α/β switch and race-free frontier updates  |
| **MPI** | distributed memory | `mpi.cpp` | 2-D -partitioned BFS that exchanges frontiers with `MPI_Allgather`/`MPI_Alltoall` and overlaps compute & comms  |
| **CUDA** | GPU | `cuda-imp.cu` | direction-optimising kernels (top-down ✕ bottom-up) with bitmap frontiers; host ↔ device traffic hidden behind persistent CSR buffers  |

All three share the same **CSR** graph layout (offsets + adjacency arrays) for cache-friendly neighbour scans and constant-time degree lookup .

---

### 2 . Directory Structure  

```
├── makefile          # builds all three flavours in one shot
├── openmp.cpp        # OpenMP + serial verifier + micro-benchmarks
├── mpi.cpp           # 2-D grid BFS for p = r×r ranks
├── cuda-imp.cu       # reference CUDA kernels (forward+backward)
└── README.md         # this file
```

---

### 3 . Algorithmic Design  

#### 3.1 Shared-memory (OpenMP)  
1. **Frontier list** (`current_frontier`) stored as a vector of vertices.  
2. Each level launches an `#pragma omp parallel` region; threads claim vertices in a *guided* schedule to limit load imbalance.  
3. Each thread buffers newly discovered neighbours in a **thread-local queue**, then merges them with `next_frontier` under a single critical section to avoid duplicates .  
4. When the ratio `edges_in_frontier / total_unvisited_edges` exceeds **α = 6**, the traversal flips to the **bottom-up** kernel that scans *unvisited* vertices and tests whether any in-edge comes from the frontier; when it falls below **β = 24** it returns to top-down .  

#### 3.2 Distributed-memory (MPI)  

*Processors form an r × r logical grid.*  
- Each rank owns a square **Aᵢⱼ** block of the adjacency matrix plus matching slices of the frontier **Fᵢⱼ**, parent mask **Pᵢⱼ**, and scratch **Tᵢⱼ**.  
- At every level we:  
  1. Symmetrically *swap* frontiers for the upper/lower triangle so both halves of **A** see the same input;  
  2. `MPI_Allgather` the column’s frontiers;  
  3. Perform a local **matrix-vector product** `Aᵢⱼ × Fⱼ` to mark neighbours;  
  4. `MPI_Alltoall` the partial results along rows;  
  5. Mask out already visited vertices and build the next frontier .  

This 2-D scheme reduces global communication volume by ≈42 % versus naïve 1-D partitioning .

#### 3.3 GPU (CUDA)  

*Direction-optimal* BFS is ported almost verbatim from Beamer et al. A compile-time tuned kernel launches one thread per vertex and toggles between:  

- **Top-down kernel**: frontier vertices iterate through outgoing edges.  
- **Bottom-up kernel**: unvisited vertices iterate through *incoming* edges and early-exit on first hit.  

Both kernels update bitmask frontiers with atomic operations and reuse a single CSR copy resident in **global memory** the entire run, so PCIe traffic is negligible (< 5 %) .

---

### 4 . Building  

```bash
# Toolchain requirements
#  * GCC 9+ (OpenMP 4.5), MPICH/OpenMPI 4+, CUDA 11.8+, Make
#
# Top-level targets are defined in makefile:
make            # builds mpi_bfs, omp_bfs, cuda_bfs
make mpi_bfs    # build just MPI variant
make clean      # remove all binaries
```

The makefile exposes variables (`CXX`, `MPICXX`, `NVCC`, optimisation flags) so you can override them on the CLI or via environment to match your architecture.

---
### 5 . Running (with the new `make` helpers)

This makefile defines three wrapper targets that both **compile and launch** the corresponding binary for you.  
Each wrapper passes any extra environment variables or command-line arguments straight through to the program.

| Task | Default parallelism | How to override | Example |
|------|--------------------|-----------------|---------|
| **MPI BFS** | `mpirun -np 4` | `NP=<procs>` | `make run_mpi NP=16 ARGS="1000000 0"` |
| **OpenMP BFS** | uses all visible threads | `OMP_NUM_THREADS=<n>` | `make run_omp ARGS="graph.txt depth.txt 0"` |
| **CUDA BFS** | first visible GPU | `CUDA_VISIBLE_DEVICES=<id>` | `make run_cuda ARGS="graph.csr 0"` |

<summary>Command templates</summary>

```bash
# MPI variant ----------------------------------------------------------
make run_mpi NP=9 ARGS="vertex_count source_id"

# OpenMP variant -------------------------------------------------------
OMP_NUM_THREADS=32 make run_omp ARGS="edge_list.txt depth_labels.txt 0"

# CUDA variant ---------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 make run_cuda ARGS="graph.csr 0"
```
---

### 6 . Benchmarking & Validation  

* **Correctness** — OpenMP driver cross-checks its results against the serial baseline; MPI prints the total number of BFS levels and can optionally dump the parent array.  
* **Timing** — MPI uses `MPI_Wtime`, OpenMP uses `omp_get_wtime`, CUDA uses `cudaEvent_t` to exclude H2D copy time (already < 5 %).  
* **Metrics** — primary metric is traversal rate (TEPS). The accompanying report shows 19.7 × speed-up on a Tesla V100 for the Email-Eu-core graph .

---

