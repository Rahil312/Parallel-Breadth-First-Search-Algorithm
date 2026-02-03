#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <fstream>
#include <string>

using namespace std;

#define cudaCheck(error) \
  if (error != cudaSuccess) { \
    printf("CUDA Error: %s at %s:%d\n", \
      cudaGetErrorString(error), \
      __FILE__, __LINE__); \
    exit(1); \
  }

// Structure to represent a graph in CSR format
struct Graph {
    unsigned int* row_ptr;    // Offsets for each vertex
    int* col_ind;            // Column indices (adjacency list)
    int nov;                // Number of vertices
};

// Direction-optimizing BFS kernel (Top-down approach)
__global__ void topDownBFSKernel(int *distance, unsigned int *row_ptr, int *col_ind, 
                               int nov, int *frontier, int *new_frontier, 
                               int *frontier_size, int *new_frontier_size, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < *frontier_size) {
        int vertex = frontier[tid];
        
        // Explore all neighbors of this vertex
        for (int edge = row_ptr[vertex]; edge < row_ptr[vertex + 1]; edge++) {
            int neighbor = col_ind[edge];
            
            // If this neighbor hasn't been visited yet
            if (distance[neighbor] == -1) {
                // Atomically update to avoid race conditions
                int old = atomicCAS(&distance[neighbor], -1, level);
                
                if (old == -1) {
                    // Add to new frontier
                    int idx = atomicAdd(new_frontier_size, 1);
                    new_frontier[idx] = neighbor;
                }
            }
        }
    }
}

// Direction-optimizing BFS kernel (Bottom-up approach)
__global__ void bottomUpBFSKernel(int *distance, unsigned int *row_ptr_inv, 
                                int *col_ind_inv, int nov, int *new_frontier, 
                                int *new_frontier_size, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nov && distance[tid] == -1) {
        // Check if any incoming neighbor is in the current frontier
        for (int edge = row_ptr_inv[tid]; edge < row_ptr_inv[tid + 1]; edge++) {
            int neighbor = col_ind_inv[edge];
            
            if (distance[neighbor] == level - 1) {
                distance[tid] = level;
                
                // Add to new frontier
                int idx = atomicAdd(new_frontier_size, 1);
                new_frontier[idx] = tid;
                break;
            }
        }
    }
}

// Kernel to count the current frontier size for metrics
__global__ void countFrontierKernel(int *distance, int nov, int level, int *count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nov && distance[tid] == level) {
        atomicAdd(count, 1);
    }
}

// Host function to create the transpose of a graph (for bottom-up approach)
void createTransposeGraph(Graph &graph, unsigned int **row_ptr_inv, int **col_ind_inv) {
    int nov = graph.nov;
    unsigned int *row_ptr = graph.row_ptr;
    int *col_ind = graph.col_ind;
    int edges = row_ptr[nov];
    
    // Allocate memory for inverse graph
    *row_ptr_inv = new unsigned int[nov + 1];
    *col_ind_inv = new int[edges];
    
    // Initialize row_ptr_inv with zeros
    for (int i = 0; i <= nov; i++) {
        (*row_ptr_inv)[i] = 0;
    }
    
    // Count incoming edges for each vertex
    for (int src = 0; src < nov; src++) {
        for (unsigned int e = row_ptr[src]; e < row_ptr[src + 1]; e++) {
            int dst = col_ind[e];
            (*row_ptr_inv)[dst + 1]++;
        }
    }
    
    // Prefix sum to calculate offsets
    for (int i = 1; i <= nov; i++) {
        (*row_ptr_inv)[i] += (*row_ptr_inv)[i - 1];
    }
    
    // Create temporary copy of row_ptr_inv to keep track of current position
    unsigned int *temp = new unsigned int[nov + 1];
    for (int i = 0; i <= nov; i++) {
        temp[i] = (*row_ptr_inv)[i];
    }
    
    // Fill col_ind_inv
    for (int src = 0; src < nov; src++) {
        for (unsigned int e = row_ptr[src]; e < row_ptr[src + 1]; e++) {
            int dst = col_ind[e];
            (*col_ind_inv)[temp[dst]++] = src;
        }
    }
    
    delete[] temp;
}

// Direction-optimizing BFS implementation
void directionOptimizingBFS(Graph &graph, int source, int *distance, double alpha = 15.0) {
    int nov = graph.nov;
    unsigned int *row_ptr = graph.row_ptr;
    int *col_ind = graph.col_ind;
    
    // Create transpose graph for bottom-up approach
    unsigned int *row_ptr_inv;
    int *col_ind_inv;
    createTransposeGraph(graph, &row_ptr_inv, &col_ind_inv);
    
    // Initialize distances
    for (int i = 0; i < nov; i++) {
        distance[i] = -1;
    }
    distance[source] = 0;
    
    // Allocate memory for frontier arrays
    int *frontier = new int[nov];
    int *new_frontier = new int[nov];
    int frontier_size = 1;
    int new_frontier_size = 0;
    
    // Initialize frontier with source vertex
    frontier[0] = source;
    
    // Allocate device memory
    int *d_distance, *d_frontier, *d_new_frontier, *d_frontier_size, *d_new_frontier_size, *d_count;
    unsigned int *d_row_ptr, *d_row_ptr_inv;
    int *d_col_ind, *d_col_ind_inv;
    
    cudaCheck(cudaMalloc((void**)&d_distance, nov * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_frontier, nov * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_new_frontier, nov * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_frontier_size, sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_new_frontier_size, sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_count, sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_row_ptr, (nov + 1) * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&d_row_ptr_inv, (nov + 1) * sizeof(unsigned int)));
    cudaCheck(cudaMalloc((void**)&d_col_ind, row_ptr[nov] * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_col_ind_inv, row_ptr[nov] * sizeof(int)));
    
    // Copy data to device
    cudaCheck(cudaMemcpy(d_distance, distance, nov * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_frontier, frontier, nov * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_frontier_size, &frontier_size, sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, (nov + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_row_ptr_inv, row_ptr_inv, (nov + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_col_ind, col_ind, row_ptr[nov] * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_col_ind_inv, col_ind_inv, row_ptr[nov] * sizeof(int), cudaMemcpyHostToDevice));
    
    // BFS traversal
    int level = 1;
    bool use_top_down = true;
    int total_edges = row_ptr[nov];
    int unvisited_vertices = nov - 1; // All except source
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start));
    
    while (frontier_size > 0) {
        // Reset new frontier size
        new_frontier_size = 0;
        cudaCheck(cudaMemcpy(d_new_frontier_size, &new_frontier_size, sizeof(int), cudaMemcpyHostToDevice));
        
        // Count unvisited neighbors for directional optimization decision
        int frontier_edges = 0;
        int block_size = 256;
        int grid_size = (nov + block_size - 1) / block_size;
        
        if (use_top_down) {
            // Top-down approach
            grid_size = (frontier_size + block_size - 1) / block_size;
            topDownBFSKernel<<<grid_size, block_size>>>(d_distance, d_row_ptr, d_col_ind, 
                                                       nov, d_frontier, d_new_frontier, 
                                                       d_frontier_size, d_new_frontier_size, level);
        } else {
            // Bottom-up approach
            bottomUpBFSKernel<<<grid_size, block_size>>>(d_distance, d_row_ptr_inv, d_col_ind_inv, 
                                                       nov, d_new_frontier, d_new_frontier_size, level);
        }
        
        // Get the new frontier size
        cudaCheck(cudaMemcpy(&new_frontier_size, d_new_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Update unvisited vertices count
        unvisited_vertices -= new_frontier_size;
        
        // Decide which direction to use for the next iteration
        // Simple heuristic: if frontier size is larger than a threshold, switch to bottom-up
        // Switch to bottom-up when frontier reaches a threshold relative to graph size
        if (use_top_down && new_frontier_size > nov / alpha) {
            use_top_down = false;
        }
        // Switch back to top-down when frontier becomes small again
        else if (!use_top_down && new_frontier_size < nov / alpha) {
            use_top_down = true;
        }
        
        // Swap frontiers for the next iteration
        std::swap(d_frontier, d_new_frontier);
        frontier_size = new_frontier_size;
        cudaCheck(cudaMemcpy(d_frontier_size, &frontier_size, sizeof(int), cudaMemcpyHostToDevice));
        
        level++;
    }
    
    // Record end time
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));
    
    // Copy results back to host
    cudaCheck(cudaMemcpy(distance, d_distance, nov * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaCheck(cudaFree(d_distance));
    cudaCheck(cudaFree(d_frontier));
    cudaCheck(cudaFree(d_new_frontier));
    cudaCheck(cudaFree(d_frontier_size));
    cudaCheck(cudaFree(d_new_frontier_size));
    cudaCheck(cudaFree(d_count));
    cudaCheck(cudaFree(d_row_ptr));
    cudaCheck(cudaFree(d_row_ptr_inv));
    cudaCheck(cudaFree(d_col_ind));
    cudaCheck(cudaFree(d_col_ind_inv));
    
    // Free host memory
    delete[] frontier;
    delete[] new_frontier;
    delete[] row_ptr_inv;
    delete[] col_ind_inv;
    
    printf("Direction-Optimizing BFS took %.4f ms\n", elapsed_time);
}

// Main function to run the BFS algorithms
int main(int argc, char *argv[]) {
    // Parse command line arguments
    if (argc < 3) {
        printf("Usage: %s <graph_file> <source_vertex>\n", argv[0]);
        return 1;
    }
    
    const char* graph_file = argv[1];
    int source = atoi(argv[2]);
    
    // Read graph from file and convert to CSR format
    // This is a placeholder - implement your own file reader
    Graph graph;
    
    // Read graph from edge list file and convert to CSR format
bool readGraphFromFile(const char* graph_file, Graph &graph) {
    // Try opening the file
    FILE *file = fopen(graph_file, "r");
    if (file == NULL) {
        // If regular open fails, try with .txt extension
        std::string graph_file_str = std::string(graph_file);
        if (graph_file_str.find(".txt") == std::string::npos) {
            graph_file_str += ".txt";
            file = fopen(graph_file_str.c_str(), "r");
        }
        
        if (file == NULL) {
            printf("Error opening file %s\n", graph_file);
            return false;
        }
    }
    
    // Create adjacency list to handle potentially non-consecutive vertex IDs
    std::unordered_map<int, std::vector<int>> adj_list;
    
    // First pass: Read edges and build adjacency list
    int src, dst;
    int max_vertex_id = -1;
    int edge_count = 0;
    
    while (fscanf(file, "%d %d", &src, &dst) == 2) {
        adj_list[src].push_back(dst);
        max_vertex_id = std::max(max_vertex_id, src);
        max_vertex_id = std::max(max_vertex_id, dst);
        edge_count++;
    }
    
    // Number of vertices is max_vertex_id + 1 (assuming 0-indexed)
    int nov = max_vertex_id + 1;
    graph.nov = nov;
    
    // Allocate memory for CSR format
    graph.row_ptr = new unsigned int[nov + 1];
    graph.col_ind = new int[edge_count];
    
    // Initialize row_ptr with zeros
    for (int i = 0; i <= nov; i++) {
        graph.row_ptr[i] = 0;
    }
    
    // Calculate row_ptr values (prefix sum of degrees)
    for (int i = 0; i < nov; i++) {
        graph.row_ptr[i+1] = graph.row_ptr[i] + adj_list[i].size();
    }
    
    // Fill col_ind from adjacency list
    for (int i = 0; i < nov; i++) {
        for (size_t j = 0; j < adj_list[i].size(); j++) {
            graph.col_ind[graph.row_ptr[i] + j] = adj_list[i][j];
        }
    }
    
    fclose(file);
    
    printf("Graph loaded: %d vertices, %d edges\n", nov, edge_count);
    return true;
}

// Read department labels from file
bool readDepartmentLabels(const char* label_file, int* department_labels, int nov) {
    // Try opening the file
    FILE *file = fopen(label_file, "r");
    if (file == NULL) {
        // If regular open fails, try with .txt extension
        std::string label_file_str = std::string(label_file);
        if (label_file_str.find(".txt") == std::string::npos) {
            label_file_str += ".txt";
            file = fopen(label_file_str.c_str(), "r");
        }
        // Try with -department-labels suffix
        if (file == NULL && label_file_str.find("-department-labels") == std::string::npos) {
            size_t dot_pos = label_file_str.find(".txt");
            if (dot_pos != std::string::npos) {
                label_file_str = label_file_str.substr(0, dot_pos) + "-department-labels.txt";
            } else {
                label_file_str += "-department-labels.txt";
            }
            file = fopen(label_file_str.c_str(), "r");
        }
        
        if (file == NULL) {
            printf("Error opening label file %s\n", label_file);
            return false;
        }
    }
    
    // Initialize labels with -1
    for (int i = 0; i < nov; i++) {
        department_labels[i] = -1;
    }
    
    int node_id, department;
    while (fscanf(file, "%d %d", &node_id, &department) == 2) {
        if (node_id < nov) {
            department_labels[node_id] = department;
        }
    }
    
    fclose(file);
    
    // Count how many nodes have labels
    int labeled_nodes = 0;
    for (int i = 0; i < nov; i++) {
        if (department_labels[i] >= 0) {
            labeled_nodes++;
        }
    }
    
    printf("Department labels loaded: %d out of %d nodes labeled\n", labeled_nodes, nov);
    return labeled_nodes > 0;
}
    
    // Placeholder for testing
    int nov = 1005;  // From your dataset statistics
    
    // Allocate distance array
    int *distance = new int[nov];
    
    // Run BFS algorithms
    directionOptimizingBFS(graph, source, distance);
    
    // Print results
    printf("BFS from source %d completed.\n", source);
    
    // Free memory
    delete[] distance;
    
    return 0;
}