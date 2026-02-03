# ---------- configurable settings ----------
# change these if your toolchain / GPU arch differs
CXX        := g++
MPICXX     := mpic++
NVCC       := nvcc
CXXFLAGS   := -O3 -std=c++17            # common flags
OMPFLAGS   := -fopenmp                  # extra for OpenMP
NVCCFLAGS  := -O3 -std=c++17            # extra for CUDA
# ---------- binaries ----------
MPI_BIN    := mpi_bfs
OMP_BIN    := omp_bfs
CUDA_BIN   := cuda_bfs
# --------------------------------------------------

.PHONY: all clean run_mpi run_omp run_cuda

all: $(MPI_BIN) $(OMP_BIN) $(CUDA_BIN)

# ---------- compilation rules ----------
$(MPI_BIN): mpi.cpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

$(OMP_BIN): openmp.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

$(CUDA_BIN): cuda-imp.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# ---------- run rules ----------
run_mpi: $(MPI_BIN)
	@mpirun -np $(or $(NP),4) ./$< $(ARGS)

run_omp: $(OMP_BIN)
	./$< $(ARGS)

run_cuda: $(CUDA_BIN)
	./$< $(ARGS)
	
# ---------- clean rule ----------
clean:
	rm -f $(MPI_BIN) $(OMP_BIN) $(CUDA_BIN)
