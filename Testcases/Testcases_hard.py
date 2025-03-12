import json

def save_code_pairs_to_json(file_path):
    code_pairs = [
        {
            "title": "Graph Cycle Detection using DFS",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

bool dfs(int node, unordered_map<int, vector<int>>& graph, vector<int>& visited, vector<int>& recStack) {
    if (!visited[node]) {
        visited[node] = recStack[node] = 1;
        for (int neighbor : graph[node]) {
            if (!visited[neighbor] && dfs(neighbor, graph, visited, recStack))
                return true;
            else if (recStack[neighbor])
                return true;
        }
    }
    recStack[node] = 0;
    return false;
}

bool hasCycle(unordered_map<int, vector<int>>& graph, int numNodes) {
    vector<int> visited(numNodes, 0), recStack(numNodes, 0);
    for (int i = 0; i < numNodes; i++) {
        if (dfs(i, graph, visited, recStack))
            return true;
    }
    return false;
}

int main() {
    unordered_map<int, vector<int>> graph = {{0, {1}}, {1, {2}}, {2, {0}}};
    cout << (hasCycle(graph, 3) ? "Cycle detected" : "No cycle") << endl;
    return 0;
}""",
            "python_code": """
def dfs(node, graph, visited, rec_stack):
    if not visited[node]:
        visited[node] = rec_stack[node] = True
        for neighbor in graph.get(node, []):
            if not visited[neighbor] and dfs(neighbor, graph, visited, rec_stack):
                return True
            elif rec_stack[neighbor]:
                return True
    rec_stack[node] = False
    return False

def has_cycle(graph, num_nodes):
    visited = [False] * num_nodes
    rec_stack = [False] * num_nodes
    for node in range(num_nodes):
        if dfs(node, graph, visited, rec_stack):
            return True
    return False

graph = {0: [1], 1: [2], 2: [0]}
print("Cycle detected" if has_cycle(graph, 3) else "No cycle")"""
        },
        {
            "title": "Dijkstra's Algorithm for Shortest Path",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

typedef pair<int, int> pii;

vector<int> dijkstra(int start, int n, vector<vector<pii>>& graph) {
    priority_queue<pii, vector<pii>, greater<pii>> pq;
    vector<int> dist(n, INT_MAX);
    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (auto& edge : graph[u]) {
            int v = edge.first, weight = edge.second;
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

int main() {
    vector<vector<pii>> graph = {{{1, 4}, {2, 1}}, {{2, 2}, {3, 5}}, {{3, 1}}, {}};
    vector<int> distances = dijkstra(0, 4, graph);
    for (int d : distances) cout << d << " ";
    return 0;
}""",
            "python_code": """
import heapq

def dijkstra(start, n, graph):
    pq = [(0, start)]
    distances = [float('inf')] * n
    distances[start] = 0

    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > distances[u]:
            continue
        for v, weight in graph.get(u, []):
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                heapq.heappush(pq, (distances[v], v))
    return distances

graph = {0: [(1, 4), (2, 1)], 1: [(2, 2), (3, 5)], 2: [(3, 1)], 3: []}
print(dijkstra(0, 4, graph))"""
        }
    ]
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(code_pairs, json_file, indent=4)

# Save the code pairs to a JSON file
save_code_pairs_to_json("code_pairs_hard.json")
