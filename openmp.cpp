#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <omp.h>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cstring>

using namespace std;

// Graph represented as an adjacency list
class Graph {
private:
    int num_vertices;
    vector<vector<int>> adj_list;
    vector<int> department; // Department labels for each node

public:
    Graph(int vertices) : num_vertices(vertices) {
        adj_list.resize(vertices);
        department.resize(vertices, -1);
    }

    void addEdge(int src, int dest) {
        if (src < num_vertices && dest < num_vertices) {
            adj_list[src].push_back(dest);
        }
    }

    void setDepartment(int node, int dept) {
        if (node < num_vertices) {
            department[node] = dept;
        }
    }

    int getDepartment(int node) const {
        if (node < num_vertices) {
            return department[node];
        }
        return -1;
    }

    int getNumVertices() const {
        return num_vertices;
    }

    const vector<int>& getNeighbors(int vertex) const {
        return adj_list[vertex];
    }

    // Method to print graph statistics
    void printStats() const {
        int totalEdges = 0;
        int maxDegree = 0;
        int minDegree = num_vertices;
        
        for (const auto& neighbors : adj_list) {
            int degree = neighbors.size();
            totalEdges += degree;
            maxDegree = max(maxDegree, degree);
            minDegree = min(minDegree, degree);
        }
        
        cout << "Graph Statistics:" << endl;
        cout << "Vertices: " << num_vertices << endl;
        cout << "Edges: " << totalEdges << endl;
        cout << "Average Degree: " << (double)totalEdges / num_vertices << endl;
        cout << "Min Degree: " << minDegree << endl;
        cout << "Max Degree: " << maxDegree << endl;
        
        // Count departments
        unordered_map<int, int> dept_count;
        for (int d : department) {
            if (d != -1) {
                dept_count[d]++;
            }
        }
        cout << "Number of departments: " << dept_count.size() << endl;
    }
};

// Function to load graph from edge list file and department file
Graph loadGraph(const string& edgeFile, const string& deptFile) {
    // First pass to determine number of vertices
    ifstream edge_file(edgeFile);
    if (!edge_file.is_open()) {
        cerr << "Error opening edge file: " << edgeFile << endl;
        exit(1);
    }
    
    int max_vertex_id = -1;
    int src, dest;
    while (edge_file >> src >> dest) {
        max_vertex_id = max(max_vertex_id, max(src, dest));
    }
    edge_file.close();
    
    // Create graph with appropriate size (add 1 because vertex IDs are 0-based)
    Graph graph(max_vertex_id + 1);
    
    // Second pass to add edges
    edge_file.open(edgeFile);
    while (edge_file >> src >> dest) {
        graph.addEdge(src, dest);
    }
    edge_file.close();
    
    // Load department information
    ifstream dept_file(deptFile);
    if (!dept_file.is_open()) {
        cerr << "Error opening department file: " << deptFile << endl;
        exit(1);
    }
    
    int node, dept;
    while (dept_file >> node >> dept) {
        graph.setDepartment(node, dept);
    }
    dept_file.close();
    
    return graph;
}

// Standard BFS implementation for verification
vector<int> serialBFS(const Graph& graph, int source) {
    int num_vertices = graph.getNumVertices();
    vector<int> distance(num_vertices, -1);
    queue<int> q;
    
    distance[source] = 0;
    q.push(source);
    
    while (!q.empty()) {
        int vertex = q.front();
        q.pop();
        
        for (int neighbor : graph.getNeighbors(vertex)) {
            if (distance[neighbor] == -1) {
                distance[neighbor] = distance[vertex] + 1;
                q.push(neighbor);
            }
        }
    }
    
    return distance;
}

// OpenMP BFS implementation (basic)
vector<int> openMPBFS(const Graph& graph, int source) {
    int num_vertices = graph.getNumVertices();
    vector<int> distance(num_vertices, -1);
    vector<int> current_frontier;
    int level = 0;
    
    // Initialize with source
    distance[source] = 0;
    current_frontier.push_back(source);
    
    // BFS iterations
    while (!current_frontier.empty()) {
        vector<int> next_frontier;
        
        #pragma omp parallel
        {
            vector<int> thread_local_queue;
            
            #pragma omp for schedule(guided, 32)
            for (size_t i = 0; i < current_frontier.size(); i++) {
                int vertex = current_frontier[i];
                const vector<int>& neighbors = graph.getNeighbors(vertex);
                
                for (int neighbor : neighbors) {
                    if (distance[neighbor] == -1) {
                        #pragma omp critical
                        {
                            if (distance[neighbor] == -1) {  // Check again to avoid race conditions
                                distance[neighbor] = level + 1;
                                thread_local_queue.push_back(neighbor);
                            }
                        }
                    }
                }
            }
            
            #pragma omp critical
            {
                next_frontier.insert(next_frontier.end(), thread_local_queue.begin(), thread_local_queue.end());
            }
        }
        
        current_frontier = next_frontier;
        level++;
    }
    
    return distance;
}

// Direction-optimizing BFS (similar to your provided code)
vector<int> directionOptimizingBFS(const Graph& graph, int source) {
    int num_vertices = graph.getNumVertices();
    vector<int> distance(num_vertices, -1);
    vector<vector<int>> inverse_graph(num_vertices);
    
    // Build the inverse graph for bottom-up approach
    for (int i = 0; i < num_vertices; i++) {
        for (int neighbor : graph.getNeighbors(i)) {
            inverse_graph[neighbor].push_back(i);
        }
    }
    
    // Count total edges for alpha/beta calculations
    long total_edges = 0;
    for (int i = 0; i < num_vertices; i++) {
        total_edges += graph.getNeighbors(i).size();
    }
    
    // Initialize with source
    distance[source] = 0;
    
    vector<int> current_frontier = {source};
    int level = 0;
    
    // Constants for direction optimization
    double alpha = 6.0;  // Threshold for switching to bottom-up
    double beta = 24.0;  // Threshold for switching back to top-down
    
    // Variables to track frontier and unvisited vertices
    long edges_in_frontier = 0;
    long total_unvisited_edges = total_edges;
    
    while (!current_frontier.empty()) {
        // Calculate edges in frontier
        edges_in_frontier = 0;
        for (int v : current_frontier) {
            edges_in_frontier += graph.getNeighbors(v).size();
        }
        
        bool use_bottom_up = (edges_in_frontier > total_unvisited_edges / alpha);
        
        vector<int> next_frontier;
        
        if (use_bottom_up) {
            // Bottom-up approach
            vector<int> unvisited;
            for (int i = 0; i < num_vertices; i++) {
                if (distance[i] == -1) {
                    unvisited.push_back(i);
                }
            }
            
            long prev_frontier_size = current_frontier.size();
            long next_frontier_size = 0;
            
            #pragma omp parallel
            {
                vector<int> thread_local_queue;
                
                #pragma omp for schedule(guided, 32)
                for (size_t i = 0; i < unvisited.size(); i++) {
                    int vertex = unvisited[i];
                    bool found = false;
                    
                    for (int parent : inverse_graph[vertex]) {
                        if (distance[parent] == level) {
                            found = true;
                            break;
                        }
                    }
                    
                    if (found) {
                        distance[vertex] = level + 1;
                        thread_local_queue.push_back(vertex);
                    }
                }
                
                #pragma omp critical
                {
                    next_frontier.insert(next_frontier.end(), thread_local_queue.begin(), thread_local_queue.end());
                }
            }
            
            next_frontier_size = next_frontier.size();
            
            // Switch back to top-down if frontier growth is too fast
            if (next_frontier_size < prev_frontier_size || next_frontier_size < num_vertices / beta) {
                use_bottom_up = false;
            }
        }
        
        if (!use_bottom_up) {
            // Top-down approach
            #pragma omp parallel
            {
                vector<int> thread_local_queue;
                
                #pragma omp for schedule(guided, 32)
                for (size_t i = 0; i < current_frontier.size(); i++) {
                    int vertex = current_frontier[i];
                    const vector<int>& neighbors = graph.getNeighbors(vertex);
                    
                    for (int neighbor : neighbors) {
                        if (distance[neighbor] == -1) {
                            #pragma omp critical
                            {
                                if (distance[neighbor] == -1) {  // Double-check
                                    distance[neighbor] = level + 1;
                                    thread_local_queue.push_back(neighbor);
                                }
                            }
                        }
                    }
                }
                
                #pragma omp critical
                {
                    next_frontier.insert(next_frontier.end(), thread_local_queue.begin(), thread_local_queue.end());
                }
            }
            
            // Update total unvisited edges
            total_unvisited_edges -= edges_in_frontier;
        }
        
        current_frontier = next_frontier;
        level++;
    }
    
    return distance;
}

// Function to measure time and verify BFS
void measureBFS(const Graph& graph, int source) {
    cout << "\nRunning BFS from source " << source << ":" << endl;

    // Serial BFS (for reference)
    double start = omp_get_wtime();
    vector<int> serial_distances = serialBFS(graph, source);
    double end = omp_get_wtime();
    cout << "Serial BFS time: " << (end - start) << " seconds" << endl;
    
    // Calculate stats
    int max_dist = -1;
    int reachable = 0;
    for (int d : serial_distances) {
        if (d != -1) {
            reachable++;
            max_dist = max(max_dist, d);
        }
    }
    cout << "  Reachable vertices: " << reachable << "/" << graph.getNumVertices() << endl;
    cout << "  Max distance: " << max_dist << endl;
    
    // Run OpenMP BFS with varying thread counts
    for (int threads = 1; threads <= 16; threads *= 2) {
        omp_set_num_threads(threads);
        
        // Basic OpenMP BFS
        start = omp_get_wtime();
        vector<int> omp_basic_distances = openMPBFS(graph, source);
        end = omp_get_wtime();
        cout << threads << " threads basic OpenMP BFS time: " << (end - start) << " seconds" << endl;
        
        // Verify results
        bool correct = true;
        for (size_t i = 0; i < serial_distances.size(); i++) {
            if (serial_distances[i] != omp_basic_distances[i]) {
                correct = false;
                break;
            }
        }
        cout << "  Results correct: " << (correct ? "Yes" : "No") << endl;
        
        // Direction-optimizing BFS
        start = omp_get_wtime();
        vector<int> omp_dir_distances = directionOptimizingBFS(graph, source);
        end = omp_get_wtime();
        cout << threads << " threads direction-optimizing BFS time: " << (end - start) << " seconds" << endl;
        
        // Verify results
        correct = true;
        for (size_t i = 0; i < serial_distances.size(); i++) {
            if (serial_distances[i] != omp_dir_distances[i]) {
                correct = false;
                break;
            }
        }
        cout << "  Results correct: " << (correct ? "Yes" : "No") << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <edge_file> <department_file>" << endl;
        return 1;
    }
    
    string edge_file = argv[1];
    string dept_file = argv[2];
    
    cout << "Loading graph from " << edge_file << " and " << dept_file << endl;
    Graph graph = loadGraph(edge_file, dept_file);
    graph.printStats();
    
    // Prepare source nodes for BFS
    // Either use a fixed source or pick random sources
    vector<int> sources;
    if (argc > 3) {
        // Use provided source
        sources.push_back(atoi(argv[3]));
    } else {
        // Pick a few random sources
        srand(time(nullptr));
        for (int i = 0; i < 3; i++) {
            sources.push_back(rand() % graph.getNumVertices());
        }
    }
    
    // Run BFS from each source
    for (int source : sources) {
        measureBFS(graph, source);
    }
    
    return 0;
}