import json

def save_code_pairs_to_json(file_path):
    code_pairs = [
                {
            "title": "01 Merge Intervals",
            "cpp_code": """
#include <iostream>              // for input and output
#include <vector>                // for std::vector
#include <algorithm>             // for std::sort

// Function to merge overlapping intervals
std::vector<std::vector<int>> mergeIntervals(std::vector<std::vector<int>>& intervals) {
    if (intervals.empty()) return {};

    // Sort intervals based on the starting point
    std::sort(intervals.begin(), intervals.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
                  return a[0] < b[0];
              });

    std::vector<std::vector<int>> merged;
    merged.push_back(intervals[0]);

    for (const auto& current : intervals) {
        auto& lastMerged = merged.back();

        // If the current interval overlaps with the last merged interval, merge them
        if (current[0] <= lastMerged[1]) {
            lastMerged[1] = std::max(lastMerged[1], current[1]);
        } else {
            // Otherwise, add the current interval as a new merged interval
            merged.push_back(current);
        }
    }

    return merged;
}

// Example usage
int main() {
    std::vector<std::vector<int>> intervals = {{1, 3}, {2, 6}, {8, 10}, {15, 18}};
    auto result = mergeIntervals(intervals);

    // Print the result
    for (const auto& interval : result) {
        std::cout << "[" << interval[0] << ", " << interval[1] << "] ";
    }
    std::cout << std::endl;  // Output: [1, 6] [8, 10] [15, 18]
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
def merge_intervals(intervals):
    if not intervals:
        return []

    # Sort intervals based on the starting point
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]  # Start with the first interval

    for current in intervals[1:]:
        last_merged = merged[-1]
        
        # If the current interval overlaps with the last merged interval, merge them
        if current[0] <= last_merged[1]:
            last_merged[1] = max(last_merged[1], current[1])
        else:
            # Otherwise, add the current interval as a new merged interval
            merged.append(current)
    
    return merged

# Example usage
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
print(merge_intervals(intervals))  # Output: [[1, 6], [8, 10], [15, 18]]
"""
        },
                {
            "title": "02 Search in Rotated Sorted Array",
            "cpp_code": """
#include <iostream>              // for input and output
#include <vector>                // for std::vector

int searchInRotatedSortedArray(const std::vector<int>& nums, int target) {
    int left = 0;
    int right = nums.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (nums[mid] == target) {
            return mid;
        }

        // Determine which side is sorted
        if (nums[left] <= nums[mid]) {  // Left side is sorted
            if (nums[left] <= target && target < nums[mid]) {  // Target is in the sorted side
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {  // Right side is sorted
            if (nums[mid] < target && target <= nums[right]) {  // Target is in the sorted side
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    return -1;  // Target not found
}

// Example usage
int main() {
    std::vector<int> nums = {4, 5, 6, 7, 0, 1, 2};
    int target = 0;
    int index = searchInRotatedSortedArray(nums, target);
    std::cout << "Index: " << index << std::endl;  // Output: Index: 4
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
def search_in_rotated_sorted_array(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Determine which side is sorted
        if nums[left] <= nums[mid]:  # Left side is sorted
            if nums[left] <= target < nums[mid]:  # Target is in the sorted side
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right side is sorted
            if nums[mid] < target <= nums[right]:  # Target is in the sorted side
                left = mid + 1
            else:
                right = mid - 1

    return -1  # Target not found

# Example usage
nums = [4, 5, 6, 7, 0, 1, 2]
target = 0
print(search_in_rotated_sorted_array(nums, target))  # Output: 4
"""
        },
                {
            "title": "03 Count Inversions in Array",
            "cpp_code": """
#include <iostream>              // for input and output
#include <vector>                // for std::vector

int mergeAndCount(std::vector<int>& arr, std::vector<int>& temp_arr, int left, int mid, int right) {
    int i = left;    // Starting index for left subarray
    int j = mid + 1; // Starting index for right subarray
    int k = left;    // Starting index to be sorted
    int inv_count = 0;

    // Conditions are checked to ensure that i doesn't exceed mid and j doesn't exceed right
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp_arr[k++] = arr[i++];
        } else {
            temp_arr[k++] = arr[j++];
            inv_count += (mid - i + 1);
        }
    }

    // Copy the remaining elements of left subarray, if any
    while (i <= mid) {
        temp_arr[k++] = arr[i++];
    }

    // Copy the remaining elements of right subarray, if any
    while (j <= right) {
        temp_arr[k++] = arr[j++];
    }

    // Copy the sorted subarray into Original array
    for (i = left; i <= right; i++) {
        arr[i] = temp_arr[i];
    }

    return inv_count;
}

int mergeSortAndCount(std::vector<int>& arr, std::vector<int>& temp_arr, int left, int right) {
    int inv_count = 0;
    if (left < right) {
        int mid = (left + right) / 2;

        inv_count += mergeSortAndCount(arr, temp_arr, left, mid);
        inv_count += mergeSortAndCount(arr, temp_arr, mid + 1, right);
        inv_count += mergeAndCount(arr, temp_arr, left, mid, right);
    }
    return inv_count;
}

int countInversions(std::vector<int>& arr) {
    std::vector<int> temp_arr(arr.size());
    return mergeSortAndCount(arr, temp_arr, 0, arr.size() - 1);
}

// Example usage
int main() {
    std::vector<int> arr = {8, 4, 2, 1};
    std::cout << countInversions(arr) << std::endl;  // Output: 6
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
def count_inversions(arr):
    def merge_and_count(arr, temp_arr, left, mid, right):
        i = left    # Starting index for left subarray
        j = mid + 1 # Starting index for right subarray
        k = left    # Starting index to be sorted
        inv_count = 0

        # Conditions are checked to ensure that i doesn't exceed mid and j doesn't exceed right
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                i += 1
            else:
                temp_arr[k] = arr[j]
                inv_count += (mid-i + 1)
                j += 1
            k += 1

        # Copy the remaining elements of left subarray, if any
        while i <= mid:
            temp_arr[k] = arr[i]
            i += 1
            k += 1

        # Copy the remaining elements of right subarray, if any
        while j <= right:
            temp_arr[k] = arr[j]
            j += 1
            k += 1

        # Copy the sorted subarray into Original array
        for i in range(left, right + 1):
            arr[i] = temp_arr[i]

        return inv_count

    def merge_sort_and_count(arr, temp_arr, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right)//2

            inv_count += merge_sort_and_count(arr, temp_arr, left, mid)
            inv_count += merge_sort_and_count(arr, temp_arr, mid + 1, right)
            inv_count += merge_and_count(arr, temp_arr, left, mid, right)

        return inv_count

    temp_arr = [0]*len(arr)
    return merge_sort_and_count(arr, temp_arr, 0, len(arr) - 1)

# Example usage
arr = [8, 4, 2, 1]
print(count_inversions(arr))  # Output: 6
"""
        },
                {
            "title": "04 Find the Missing Number",
            "cpp_code": """
#include <iostream>              // for input and output
#include <vector>                // for std::vector

int findMissingNumber(const std::vector<int>& arr) {
    int n = arr.size();
    int expected_sum = (n * (n + 1)) / 2;
    int actual_sum = 0;

    for (int num : arr) {
        actual_sum += num;
    }

    return expected_sum - actual_sum;
}

// Example usage
int main() {
    std::vector<int> arr = {3, 0, 1};
    std::cout << findMissingNumber(arr) << std::endl;  // Output: 2
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
def find_missing_number(arr):
    n = len(arr)
    expected_sum = (n * (n + 1)) // 2
    actual_sum = sum(arr)
    return expected_sum - actual_sum

# Example usage
arr = [3, 0, 1]
print(find_missing_number(arr))  # Output: 2
"""
        },
                {
            "title": "05 Longest Increasing Subsequence",
            "cpp_code": """
#include <iostream>              // for input and output
#include <vector>                // for std::vector
#include <algorithm>             // for std::max

int lengthOfLIS(const std::vector<int>& nums) {
    if (nums.empty()) return 0;

    int n = nums.size();
    std::vector<int> dp(n, 1);

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (nums[i] > nums[j]) {
                dp[i] = std::max(dp[i], dp[j] + 1);
            }
        }
    }

    return *std::max_element(dp.begin(), dp.end());
}

// Example usage
int main() {
    std::vector<int> nums = {10, 9, 2, 5, 3, 7, 101, 18};
    std::cout << lengthOfLIS(nums) << std::endl;  // Output: 4
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
def length_of_lis(nums):
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# Example usage
nums = [10, 9, 2, 5, 3, 7, 101, 18]
print(length_of_lis(nums))  # Output: 4
"""
        },
                {
            "title": "06 Minimum Window Substring",
            "cpp_code": """
#include <iostream>              // for input and output
#include <unordered_map>         // for std::unordered_map
#include <climits>               // for INT_MAX

std::string minWindowSubstring(const std::string& S, const std::string& T) {
    if (S.empty() || T.empty()) return "";
    
    std::unordered_map<char, int> target_count;
    std::unordered_map<char, int> window_count;
    
    for (char c : T) {
        target_count[c]++;
    }
    
    int required = target_count.size();
    int formed = 0;
    
    int l = 0, r = 0;
    int min_len = INT_MAX;
    std::string min_window;
    
    while (r < S.size()) {
        char c = S[r];
        window_count[c]++;
        
        if (target_count.find(c) != target_count.end() && window_count[c] == target_count[c]) {
            formed++;
        }
        
        while (l <= r && formed == required) {
            c = S[l];
            
            if (r - l + 1 < min_len) {
                min_len = r - l + 1;
                min_window = S.substr(l, min_len);
            }
            
            window_count[c]--;
            if (target_count.find(c) != target_count.end() && window_count[c] < target_count[c]) {
                formed--;
            }
            
            l++;
        }
        
        r++;
    }
    
    return min_window;
}

// Example usage
int main() {
    std::string S = "ADOBECODEBANC";
    std::string T = "ABC";
    std::cout << minWindowSubstring(S, T) << std::endl;  // Output: "BANC"
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
from collections import Counter, defaultdict

def min_window_substring(S, T):
    if not S or not T:
        return ""
    
    target_count = Counter(T)
    window_count = defaultdict(int)
    required = len(target_count)
    formed = 0
    
    l, r = 0, 0
    min_len = float('inf')
    min_window = ""
    
    while r < len(S):
        char = S[r]
        window_count[char] += 1
        
        if char in target_count and window_count[char] == target_count[char]:
            formed += 1
        
        while l <= r and formed == required:
            char = S[l]
            
            if r - l + 1 < min_len:
                min_len = r - l + 1
                min_window = S[l:r+1]
            
            window_count[char] -= 1
            if char in target_count and window_count[char] < target_count[char]:
                formed -= 1
            
            l += 1
        
        r += 1
    
    return min_window

# Example usage
S = "ADOBECODEBANC"
T = "ABC"
print(min_window_substring(S, T))  # Output: "BANC"
"""
        },
                {
            "title": "07 Graph Coloring Problem",
            "cpp_code": """
#include <iostream>               // for input and output
#include <vector>                 // for std::vector
#include <unordered_map>          // for std::unordered_map

bool isValid(const std::unordered_map<int, std::vector<int>>& graph, const std::vector<int>& colors, int vertex, int color) {
    for (int neighbor : graph.at(vertex)) {
        if (colors[neighbor] == color) {
            return false;
        }
    }
    return true;
}

bool graphColoringUtil(const std::unordered_map<int, std::vector<int>>& graph, std::vector<int>& colors, int vertex, int M) {
    if (vertex == graph.size()) {
        return true;
    }
    
    for (int color = 1; color <= M; ++color) {
        if (isValid(graph, colors, vertex, color)) {
            colors[vertex] = color;
            if (graphColoringUtil(graph, colors, vertex + 1, M)) {
                return true;
            }
            colors[vertex] = -1;  // Backtrack
        }
    }
    
    return false;
}

bool graphColoring(const std::unordered_map<int, std::vector<int>>& graph, int M) {
    std::vector<int> colors(graph.size(), -1);
    return graphColoringUtil(graph, colors, 0, M);
}

// Example usage
int main() {
    std::unordered_map<int, std::vector<int>> graph = {
        {0, {1, 2}},
        {1, {0, 2}},
        {2, {0, 1}}
    };
    int M = 2;
    std::cout << (graphColoring(graph, M) ? "True" : "False") << std::endl;  // Output: True
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
def is_valid(graph, colors, vertex, color):
    for neighbor in graph[vertex]:
        if colors[neighbor] == color:
            return False
    return True

def graph_coloring_util(graph, colors, vertex, M):
    if vertex == len(graph):
        return True
    
    for color in range(1, M + 1):
        if is_valid(graph, colors, vertex, color):
            colors[vertex] = color
            if graph_coloring_util(graph, colors, vertex + 1, M):
                return True
            colors[vertex] = -1  # Backtrack
    
    return False

def graph_coloring(graph, M):
    colors = [-1] * len(graph)
    return graph_coloring_util(graph, colors, 0, M)

# Example usage
graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
M = 2
print(graph_coloring(graph, M))  # Output: True
"""
        },
                {
            "title": "08 Topological Sort Kahn Algorithm",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

std::vector<int> topologicalSortKahn(const std::unordered_map<int, std::vector<int>>& graph) {
    std::unordered_map<int, int> inDegree;
    for (const auto& pair : graph) {
        inDegree[pair.first] = 0;
    }
    for (const auto& pair : graph) {
        for (int neighbor : pair.second) {
            inDegree[neighbor]++;
        }
    }

    std::queue<int> queue;
    for (const auto& pair : inDegree) {
        if (pair.second == 0) {
            queue.push(pair.first);
        }
    }

    std::vector<int> topologicalOrder;
    while (!queue.empty()) {
        int node = queue.front();
        queue.pop();
        topologicalOrder.push_back(node);

        for (int neighbor : graph.at(node)) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                queue.push(neighbor);
            }
        }
    }

    return topologicalOrder;
}

// Example usage
int main() {
    std::unordered_map<int, std::vector<int>> graph = {
        {5, {2, 0}},
        {4, {0, 1}},
        {2, {3}},
        {3, {1}}
    };
    std::vector<int> result = topologicalSortKahn(graph);
    for (int node : result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;  // Output: 4 5 2 3 1 0

    return 0;
}""",
            "python_code": """
from collections import deque, defaultdict

def topological_sort_kahn(graph):
    # Step 1: Calculate in-degrees
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # Step 2: Initialize the queue with nodes having 0 in-degree
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topological_order = []
    
    # Step 3: Process nodes in the queue
    while queue:
        node = queue.popleft()
        topological_order.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if topological_order contains all nodes
    if len(topological_order) == len(graph):
        return topological_order
    else:
        return []  # Graph has a cycle or is not a DAG

# Example usage
graph = {5: [2, 0], 4: [0, 1], 2: [3], 3: [1]}
print(topological_sort_kahn(graph))  # Output: [4, 5, 2, 3, 1, 0]
"""
        },
        {
            "title": "09 Topological Sort DFS Based Approach",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stack>

void dfs(int node, const std::unordered_map<int, std::vector<int>>& graph, std::unordered_set<int>& visited, std::stack<int>& Stack) {
    visited.insert(node);
    for (int neighbor : graph.at(node)) {
        if (visited.find(neighbor) == visited.end()) {
            dfs(neighbor, graph, visited, Stack);
        }
    }
    Stack.push(node);
}

std::vector<int> topologicalSortDFS(const std::unordered_map<int, std::vector<int>>& graph) {
    std::unordered_set<int> visited;
    std::stack<int> Stack;
    
    for (const auto& pair : graph) {
        if (visited.find(pair.first) == visited.end()) {
            dfs(pair.first, graph, visited, Stack);
        }
    }
    
    std::vector<int> topologicalOrder;
    while (!Stack.empty()) {
        topologicalOrder.push_back(Stack.top());
        Stack.pop();
    }
    
    return topologicalOrder;
}

// Example usage
int main() {
    std::unordered_map<int, std::vector<int>> graph = {
        {5, {2, 0}},
        {4, {0, 1}},
        {2, {3}},
        {3, {1}}
    };
    std::vector<int> result = topologicalSortDFS(graph);
    for (int node : result) {
        std::cout << node << " ";
    }
    std::cout << std::endl;  // Output: 4 5 2 3 1 0

    return 0;
}""",
            "python_code": """
def topological_sort_dfs(graph):
    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        topological_order.append(node)
    
    visited = set()
    topological_order = []
    
    # Visit all nodes
    for node in graph:
        if node not in visited:
            dfs(node)
    
    return topological_order[::-1]  # Reverse to get the correct order

# Example usage
graph = {5: [2, 0], 4: [0, 1], 2: [3], 3: [1]}
print(topological_sort_dfs(graph))  # Output: [4, 5, 2, 3, 1, 0]
"""
        },
{
            "title": "10 Matrix Chain Multiplication",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <climits>

int matrixChainMultiplication(const std::vector<int>& dimensions) {
    int n = dimensions.size() - 1;
    std::vector<std::vector<int>> m(n, std::vector<int>(n, 0));
    
    // l is the chain length
    for (int l = 2; l <= n; ++l) {  // l ranges from 2 to n
        for (int i = 0; i < n - l + 1; ++i) {
            int j = i + l - 1;
            m[i][j] = INT_MAX;
            for (int k = i; k < j; ++k) {
                int q = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1];
                if (q < m[i][j]) {
                    m[i][j] = q;
                }
            }
        }
    }
    
    return m[0][n-1];
}

// Example usage
int main() {
    std::vector<int> dimensions = {10, 20, 30, 40, 30};
    std::cout << "Minimum number of multiplications: " << matrixChainMultiplication(dimensions) << std::endl;  // Output: 30000
    return 0;
}""",
            "python_code": """
def matrix_chain_multiplication(dimensions):
    n = len(dimensions) - 1
    # m[i][j] will be the minimum number of multiplications needed to compute the product of matrices from i to j
    m = [[0] * n for _ in range(n)]
    
    # l is the chain length
    for l in range(2, n+1):  # l ranges from 2 to n
        for i in range(n - l + 1):
            j = i + l - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k+1][j] + dimensions[i] * dimensions[k+1] * dimensions[j+1]
                if q < m[i][j]:
                    m[i][j] = q
    
    return m[0][n-1]

# Example usage
dimensions = [10, 20, 30, 40, 30]
print("Minimum number of multiplications:", matrix_chain_multiplication(dimensions))  # Output: 30000
"""
        },
                {
            "title": "11 Travelling Salesman Problem",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <climits>

int tsp(const std::vector<std::vector<int>>& distance) {
    int n = distance.size();
    std::vector<std::vector<int>> dp(1 << n, std::vector<int>(n, INT_MAX));
    dp[1][0] = 0;  // Start at city 0

    for (int mask = 1; mask < (1 << n); ++mask) {
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) {
                for (int j = 0; j < n; ++j) {
                    if (!(mask & (1 << j))) {
                        int new_mask = mask | (1 << j);
                        dp[new_mask][j] = std::min(dp[new_mask][j], dp[mask][i] + distance[i][j]);
                    }
                }
            }
        }
    }

    // Compute the minimum cost to return to the starting city (0)
    int result = INT_MAX;
    for (int i = 1; i < n; ++i) {
        result = std::min(result, dp[(1 << n) - 1][i] + distance[i][0]);
    }

    return result;
}

int main() {
    std::vector<std::vector<int>> distance_matrix = {{0, 10, 15, 20}, {10, 0, 35, 25}, {15, 35, 0, 30}, {20, 25, 30, 0}};
    std::cout << "Minimum cost: " << tsp(distance_matrix) << std::endl;  // Output: 80
    return 0;
}""",
            "python_code": """
def tsp(distance):
    n = len(distance)
    # Initialize the dp array with infinity
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at city 0

    for mask in range(1 << n):
        for i in range(n):
            if mask & (1 << i):
                for j in range(n):
                    if mask & (1 << j) == 0:
                        new_mask = mask | (1 << j)
                        dp[new_mask][j] = min(dp[new_mask][j], dp[mask][i] + distance[i][j])

    # Compute the minimum cost to return to the starting city (0)
    result = min(dp[(1 << n) - 1][i] + distance[i][0] for i in range(1, n))
    
    return result

# Example usage
distance_matrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
print("Minimum cost:", tsp(distance_matrix))  # Output: 80
"""
        }
    ]
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(code_pairs, json_file, indent=4)

# Save the code pairs to a JSON file
save_code_pairs_to_json("Hand_testcases_AlgorithmicChallenges.json")
