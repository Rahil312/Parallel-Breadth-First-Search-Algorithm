#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

/**
 * Checks if there are any vertices in the frontier to process
 * Returns true if at least one vertex is in the frontier (has value 1)
 */
bool isFrontierActive(int F[], long int size) {
    for (int i = 0; i < size; i++) {
        if (F[i] == 1) {
            return true;
        }
    }
    return false;
}

/**
 * Retrieves linearized index for 2D matrix access
 */
long long int getIndex(long long int i, long long int j, long long int rowSize) {
    return i * rowSize + j;
}

int main(int argc, char *argv[]) {
    // Initialize variables for graph processing
    unsigned long long int noOfVertices;        // Total number of vertices in the graph
    unsigned long long int rowNo, columnNo;     // Position of processor in the 2D grid
    unsigned long long int noOfPRows;           // Number of processor rows in the 2D grid
    unsigned long long int verticesPerProc;     // Number of vertices owned by each processor
    unsigned long long int source;              // Source vertex for BFS
    
    int rank, numProcessors;
    double startTime, endTime;
    MPI_Status status;
    
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    startTime = MPI_Wtime();  // Start timing
    
    // Validate command line arguments
    if (argc < 3) {
        if (rank == 0) {
            printf("Usage: %s <num_vertices> <source_vertex> [<input_file>]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Parse command line arguments
    noOfVertices = atoll(argv[1]);
    source = atoll(argv[2]);
    
    // Validate source vertex
    if (source >= noOfVertices) {
        if (rank == 0) {
            printf("Error: Source vertex %llu is out of range (0-%llu)\n", source, noOfVertices-1);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Calculate 2D processor grid dimensions
    noOfPRows = (int)sqrt(numProcessors);
    if (noOfPRows * noOfPRows != numProcessors) {
        if (rank == 0) {
            printf("Error: Number of processors must be a perfect square\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Calculate processor coordinates in the 2D grid
    rowNo = rank / noOfPRows + 1;
    columnNo = rank % noOfPRows + 1;
    
    // Calculate how many vertices each processor owns
    verticesPerProc = (int)ceil(noOfVertices / (double)(noOfPRows * noOfPRows));
    unsigned long long int verticesInProcRow = verticesPerProc * noOfPRows;
    
    if (rank == 0) {
        printf("Initializing BFS from source vertex %llu on a graph with %llu vertices\n", 
               source, noOfVertices);
        printf("Using %d processors in a %llu x %llu grid\n", 
               numProcessors, noOfPRows, noOfPRows);
        printf("Each processor manages %llu vertices\n", verticesPerProc);
    }
    
    // Initialize data structures for BFS
    // F: Global frontier vector (tracks vertices to visit in current level)
    int* F = (int*)malloc(sizeof(int) * noOfVertices);
    for (unsigned long long int i = 0; i < noOfVertices; i++) {
        F[i] = 0;
    }
    F[source] = 1;  // Mark source vertex as frontier
    
    // Fij: Local frontier for this processor
    int* Fij = (int*)malloc(sizeof(int) * verticesPerProc);
    
    // Pij: Parent/visited vector (tracks vertices already visited)
    int* Pij = (int*)malloc(sizeof(int) * verticesPerProc);
    
    // Tij: Next level frontier for this processor
    int* Tij = (int*)malloc(sizeof(int) * verticesPerProc);
    
    for (unsigned long long int i = 0; i < verticesPerProc; i++) {
        Fij[i] = 0;
        Pij[i] = 0;
        Tij[i] = 0;
    }
    
    // Ti: Temporary vector to hold combined results from a row of processors
    int* Ti = (int*)malloc(sizeof(int) * verticesInProcRow);
    for (unsigned long long int i = 0; i < verticesInProcRow; i++) {
        Ti[i] = 0;
    }
    
    // Aij: Local adjacency matrix for this processor
    char* Aij = (char*)malloc(sizeof(char) * verticesInProcRow * verticesInProcRow);
    for (unsigned long long int i = 0; i < verticesInProcRow * verticesInProcRow; i++) {
        Aij[i] = 0;
    }
    
    // Create row and column communicators for the 2D processor grid
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(MPI_COMM_WORLD, rowNo, rank, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, columnNo, rank, &colComm);
    
    // Read graph data or generate random graph
    if (argc > 4) {  // We need two files: edge list and department labels
        if (rank == 0) {
            printf("Reading graph from edge list file %s and labels from %s\n", argv[3], argv[4]);
        }
        
        // First file contains edge list (b.txt in your example)
        FILE *edgeFile = NULL;
        if (rank == 0) {
            edgeFile = fopen(argv[3], "r");
            if (!edgeFile) {
                printf("Error: Cannot open edge list file %s\n", argv[3]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // Read edge list and build global adjacency matrix
            int src, dst;
            char *globalAdjMatrix = (char*)calloc(noOfVertices * noOfVertices, sizeof(char));
            
            while (fscanf(edgeFile, "%d %d", &src, &dst) == 2) {
                if (src < noOfVertices && dst < noOfVertices) {
                    globalAdjMatrix[src * noOfVertices + dst] = 1;
                }
            }
            fclose(edgeFile);
            
            // Distribute the adjacency matrix to all processors
            for (int p = 0; p < numProcessors; p++) {
                int procRow = p / noOfPRows;
                int procCol = p % noOfPRows;
                
                // Calculate which part of the global matrix this processor needs
                for (unsigned long long int i = 0; i < verticesPerProc; i++) {
                    unsigned long long int globalRow = procRow * verticesPerProc + i;
                    if (globalRow >= noOfVertices) continue;
                    
                    for (unsigned long long int j = 0; j < verticesPerProc; j++) {
                        unsigned long long int globalCol = procCol * verticesPerProc + j;
                        if (globalCol >= noOfVertices) continue;
                        
                        char value = globalAdjMatrix[globalRow * noOfVertices + globalCol];
                        
                        if (p == rank) {
                            // This is our processor, store locally
                            Aij[i * verticesInProcRow + j] = value;
                        } else {
                            // Send to appropriate processor
                            MPI_Send(&value, 1, MPI_CHAR, p, 0, MPI_COMM_WORLD);
                        }
                    }
                }
            }
            
            free(globalAdjMatrix);
        } else {
            // Receive my portion of the adjacency matrix
            for (unsigned long long int i = 0; i < verticesPerProc; i++) {
                unsigned long long int globalRow = (rank / noOfPRows) * verticesPerProc + i;
                if (globalRow >= noOfVertices) continue;
                
                for (unsigned long long int j = 0; j < verticesPerProc; j++) {
                    unsigned long long int globalCol = (rank % noOfPRows) * verticesPerProc + j;
                    if (globalCol >= noOfVertices) continue;
                    
                    char value;
                    MPI_Recv(&value, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
                    Aij[i * verticesInProcRow + j] = value;
                }
            }
        }
        
        // Second file contains department labels
        // We could use this for community detection or visualization
        int *departmentLabels = NULL;
        if (rank == 0) {
            departmentLabels = (int*)malloc(sizeof(int) * noOfVertices);
            FILE *labelFile = fopen(argv[4], "r");
            if (!labelFile) {
                printf("Warning: Cannot open department labels file %s\n", argv[4]);
            } else {
                int node, dept;
                while (fscanf(labelFile, "%d %d", &node, &dept) == 2) {
                    if (node < noOfVertices) {
                        departmentLabels[node] = dept;
                    }
                }
                fclose(labelFile);
                printf("Department labels loaded\n");
            }
            free(departmentLabels); // Free as we don't use them for BFS
        }
        
    } else {
        // Generate random graph (for testing)
        if (rank == 0) {
            printf("Generating random graph with 10%% density\n");
        }
        
        srand(100 * rank);
        for (unsigned long long int i = 0; i < verticesInProcRow * verticesInProcRow; i++) {
            if (rand() % 100 < 10) { // 10% density
                Aij[i] = 1;
            }
        }
    }
    
    // Initialize local frontier based on global frontier
    for (unsigned long long int i = 0; i < verticesPerProc; i++) {
        unsigned long long int globalIndex = (rowNo - 1) * verticesPerProc * noOfPRows + 
                                           (columnNo - 1) * verticesPerProc + i;
        
        if (globalIndex < noOfVertices) {
            Fij[i] = F[globalIndex];
            Pij[i] = Fij[i]; // Mark source as visited
        }
    }
    
    // Create buffers for communication
    int* recvBuffer = (int*)malloc(sizeof(int) * verticesInProcRow);
    int* sendBuffer = (int*)malloc(sizeof(int) * verticesPerProc);
    
    // Mark end of initialization phase
    double initTime = MPI_Wtime();
    if (rank == 0) {
        printf("Initialization completed in %f seconds\n", initTime - startTime);
    }
    
    // Main BFS algorithm
    int level = 0;
    while (isFrontierActive(F, noOfVertices)) {
        level++;
        if (rank == 0) {
            printf("BFS Level %d\n", level);
        }
        
        // 1. Share symmetric parts of the frontier (for processors not on diagonal)
        for (unsigned long long int i = 0; i < verticesPerProc; i++) {
            sendBuffer[i] = Fij[i];
        }
        
        if (rowNo != columnNo) {
            if (rowNo > columnNo) {
                MPI_Send(sendBuffer, verticesPerProc, MPI_INT, 
                        (columnNo - 1) * noOfPRows + rowNo - 1, 123, MPI_COMM_WORLD);
                MPI_Recv(Fij, verticesPerProc, MPI_INT, 
                        (columnNo - 1) * noOfPRows + rowNo - 1, 123, MPI_COMM_WORLD, &status);
            } else {
                MPI_Recv(Fij, verticesPerProc, MPI_INT, 
                        (columnNo - 1) * noOfPRows + rowNo - 1, 123, MPI_COMM_WORLD, &status);
                MPI_Send(sendBuffer, verticesPerProc, MPI_INT, 
                        (columnNo - 1) * noOfPRows + rowNo - 1, 123, MPI_COMM_WORLD);
            }
        }
        
        // 2. Gather frontier from all processors in the same column
        MPI_Allgather(Fij, verticesPerProc, MPI_INT, 
                     recvBuffer, verticesPerProc, MPI_INT, colComm);
        
        // 3. Matrix-vector multiplication: find neighbors of frontier vertices
        for (unsigned long long int i = 0; i < verticesInProcRow; i++) {
            int val = 0;
            for (unsigned long long int j = 0; j < verticesInProcRow; j++) {
                val += Aij[getIndex(i, j, verticesInProcRow)] * recvBuffer[j];
            }
            Ti[i] = val;
        }
        
        // 4. Exchange partial results with processors in the same row
        MPI_Alltoall(Ti, verticesPerProc, MPI_INT, 
                    recvBuffer, verticesPerProc, MPI_INT, rowComm);
        
        // 5. Process received results to update frontier
        for (unsigned long long int i = 0; i < noOfPRows; i++) {
            for (unsigned long long int j = 0; j < verticesPerProc; j++) {
                if (recvBuffer[i * verticesPerProc + j] > 0) {
                    Tij[j] = 1;
                }
            }
        }
        
        // 6. Update frontier (remove already visited vertices)
        for (unsigned long long int i = 0; i < verticesPerProc; i++) {
            if (Pij[i] == 1 && Tij[i] == 1) {
                Tij[i] = 0;  // Remove already visited vertices
            }
        }
        
        // 7. Update parent/visited vector
        for (unsigned long long int i = 0; i < verticesPerProc; i++) {
            if (Pij[i] == 0 && Tij[i] == 1) {
                Pij[i] = 1;  // Mark newly discovered vertices as visited
            }
        }
        
        // 8. Update local frontier for next iteration
        for (unsigned long long int i = 0; i < verticesPerProc; i++) {
            Fij[i] = Tij[i];
            Tij[i] = 0;  // Reset Tij for next iteration
        }
        
        // 9. Gather local frontiers to build global frontier
        MPI_Allgather(Fij, verticesPerProc, MPI_INT, 
                     recvBuffer, verticesPerProc, MPI_INT, rowComm);
        MPI_Allgather(recvBuffer, verticesInProcRow, MPI_INT, 
                     F, verticesInProcRow, MPI_INT, colComm);
    }
    
    // Report timing results
    endTime = MPI_Wtime();
    double localTime = endTime - initTime;
    double globalTime;
    
    MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("BFS traversal completed in %f seconds (%d levels)\n", globalTime, level);
    }
    
    // Clean up
    free(F);
    free(Fij);
    free(Pij);
    free(Tij);
    free(Ti);
    free(Aij);
    free(recvBuffer);
    free(sendBuffer);
    
    MPI_Finalize();
    return 0;
}