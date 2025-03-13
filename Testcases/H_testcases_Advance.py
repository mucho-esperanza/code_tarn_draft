import json

def save_code_pairs_to_json(file_path):
    code_pairs = [
        {
            "title": "01 Sudoku Solver",
            "cpp_code": """
#include <iostream>
#include <vector>

using namespace std;

bool is_valid(vector<vector<int>>& board, int row, int col, int num) {
    for (int i = 0; i < 9; ++i) {
        if (board[row][i] == num || board[i][col] == num) {
            return false;
        }
        if (board[row / 3 * 3 + i / 3][col / 3 * 3 + i % 3] == num) {
            return false;
        }
    }
    return true;
}

bool solve_sudoku(vector<vector<int>>& board) {
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (board[i][j] == 0) {
                for (int num = 1; num <= 9; ++num) {
                    if (is_valid(board, i, j, num)) {
                        board[i][j] = num;
                        if (solve_sudoku(board)) {
                            return true;
                        }
                        board[i][j] = 0;
                    }
                }
                return false;
            }
        }
    }
    return true;
}

int main() {
    vector<vector<int>> sudoku_board = {
        {5, 3, 0, 0, 7, 0, 0, 0, 0},
        {6, 0, 0, 1, 9, 5, 0, 0, 0},
        {0, 9, 8, 0, 0, 0, 0, 6, 0},
        {8, 0, 0, 0, 6, 0, 0, 0, 3},
        {4, 0, 0, 8, 0, 3, 0, 0, 1},
        {7, 0, 0, 0, 2, 0, 0, 0, 6},
        {0, 6, 0, 0, 0, 0, 2, 8, 0},
        {0, 0, 0, 4, 1, 9, 0, 0, 5},
        {0, 0, 0, 0, 8, 0, 0, 7, 9}
    };

    if (solve_sudoku(sudoku_board)) {
        for (const auto& row : sudoku_board) {
            for (int num : row) {
                cout << num << " ";
            }
            cout << endl;
        }
    } else {
        cout << "No solution exists!" << endl;
    }

    return 0;
}""",
            "python_code": """
def is_valid(board, row, col, num):
    # Check if 'num' is not in the current row, column, and 3x3 subgrid
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
        if board[row // 3 * 3 + i // 3][col // 3 * 3 + i % 3] == num:
            return False
    return True

def solve_sudoku(board):
    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = 0
                    return False
        return True

    backtrack()
    return board

# Example usage
sudoku_board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

solved_board = solve_sudoku(sudoku_board)
for row in solved_board:
    print(row)
"""
        },
        {
            "title": "02 Knapsack Problem",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int knapsack(vector<int>& weights, vector<int>& values, int capacity) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));

    for (int i = 1; i <= n; ++i) {
        for (int w = 0; w <= capacity; ++w) {
            if (weights[i - 1] <= w) {
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w]);
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    return dp[n][capacity];
}

int main() {
    int n, capacity;
    n = 3;

    vector<int> weights(n);
    vector<int> values(n);

    cout << "Enter the weights of the items:\n";
    for (int i = 0; i < n; ++i) {
        cin >> weights[i];
    }

    cout << "Enter the values of the items:\n";
    for (int i = 0; i < n; ++i) {
        cin >> values[i];
    }

    cout << "Enter the capacity of the knapsack: ";
    cin >> capacity;

    int max_value = knapsack(weights, values, capacity);
    cout << "Maximum Value: " << max_value << endl;

    return 0;
}""",
            "python_code": """
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]

# Example usage
weights = [1, 3, 4]
values = [15, 20, 30]
capacity = 4

max_value = knapsack(weights, values, capacity)
print("Maximum Value:", max_value)
"""
        },
        {
            "title": "03 LRU Cache Implementation",
            "cpp_code": """
#include <iostream>
#include <unordered_map>

using namespace std;

class Node {
public:
    int key;
    int value;
    Node* prev;
    Node* next;
    Node(int k, int v) : key(k), value(v), prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    int capacity;
    unordered_map<int, Node*> cache;
    Node* head;
    Node* tail;

    void remove(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void add(Node* node) {
        node->prev = tail->prev;
        node->next = tail;
        tail->prev->next = node;
        tail->prev = node;
    }

public:
    LRUCache(int cap) : capacity(cap) {
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head->next = tail;
        tail->prev = head;
    }

    int get(int key) {
        if (cache.find(key) != cache.end()) {
            Node* node = cache[key];
            remove(node);
            add(node);
            return node->value;
        }
        return -1;
    }

    void put(int key, int value) {
        if (cache.find(key) != cache.end()) {
            remove(cache[key]);
            delete cache[key];
        }
        Node* node = new Node(key, value);
        add(node);
        cache[key] = node;
        if (cache.size() > capacity) {
            Node* lru = head->next;
            remove(lru);
            cache.erase(lru->key);
            delete lru;
        }
    }
};

// Example usage
int main() {
    LRUCache cache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    cout << cache.get(1) << endl;  // Output: 1
    cache.put(3, 3);
    cout << cache.get(2) << endl;  // Output: -1
    return 0;
}""",
            "python_code": """
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        prev = node.prev
        next = node.next
        prev.next = next
        next.prev = prev

    def _add(self, node):
        prev = self.tail.prev
        prev.next = node
        node.prev = prev
        node.next = self.tail
        self.tail.prev = node

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]

# Example usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # Output: 1
cache.put(3, 3)
print(cache.get(2))  # Output: -1
"""
        },
        {
            "title": "04 Word Ladder",
            "cpp_code": """
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>

using namespace std;

vector<string> wordLadder(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (wordSet.find(endWord) == wordSet.end()) {
        return {};
    }
    
    unordered_map<string, vector<string>> allComboDict;
    int L = beginWord.length();
    
    for (const string& word : wordList) {
        for (int i = 0; i < L; ++i) {
            string newWord = word.substr(0, i) + '*' + word.substr(i + 1, L - i - 1);
            allComboDict[newWord].push_back(word);
        }
    }
    
    queue<pair<string, vector<string>>> q;
    q.push({beginWord, {beginWord}});
    unordered_set<string> visited = {beginWord};
    
    while (!q.empty()) {
        auto [currentWord, path] = q.front();
        q.pop();
        
        for (int i = 0; i < L; ++i) {
            string intermediateWord = currentWord.substr(0, i) + '*' + currentWord.substr(i + 1, L - i - 1);
            
            for (const string& word : allComboDict[intermediateWord]) {
                if (word == endWord) {
                    path.push_back(endWord);
                    return path;
                }
                
                if (visited.find(word) == visited.end()) {
                    visited.insert(word);
                    vector<string> newPath = path;
                    newPath.push_back(word);
                    q.push({word, newPath});
                }
            }
        }
    }
    
    return {};
}

int main() {
    string beginWord = "hit";
    string endWord = "cog";
    vector<string> wordList = {"hot", "dot", "dog", "lot", "log", "cog"};
    vector<string> result = wordLadder(beginWord, endWord, wordList);
    
    for (const string& word : result) {
        cout << word << " ";
    }
    // Output: hit hot dot dog cog
    return 0;
}""",
            "python_code": """
from collections import deque, defaultdict

def word_ladder(begin_word, end_word, word_list):
    if end_word not in word_list:
        return []

    # Preprocess word list to create a dictionary of possible transformations
    word_list.append(begin_word)
    all_combo_dict = defaultdict(list)
    L = len(begin_word)
    
    for word in word_list:
        for i in range(L):
            all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)

    # BFS initialization
    queue = deque([(begin_word, [begin_word])])
    visited = set([begin_word])

    while queue:
        current_word, path = queue.popleft()
        for i in range(L):
            intermediate_word = current_word[:i] + "*" + current_word[i+1:]

            for word in all_combo_dict[intermediate_word]:
                if word == end_word:
                    return path + [end_word]
                
                if word not in visited:
                    visited.add(word)
                    queue.append((word, path + [word]))
    return []

# Example usage
begin_word = "hit"
end_word = "cog"
word_list = ["hot", "dot", "dog", "lot", "log", "cog"]
print(word_ladder(begin_word, end_word, word_list))  # Output: ['hit', 'hot', 'dot', 'dog', 'cog']
"""
        },
        {
            "title": "05 K-th Largest Element",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

int partition(vector<int>& nums, int left, int right) {
    int pivot_index = left + rand() % (right - left + 1);
    swap(nums[pivot_index], nums[right]);
    int pivot = nums[right];
    int i = left;
    for (int j = left; j < right; ++j) {
        if (nums[j] <= pivot) {
            swap(nums[i], nums[j]);
            ++i;
        }
    }
    swap(nums[i], nums[right]);
    return i;
}

int quickselect(vector<int>& nums, int left, int right, int k) {
    if (left == right) {
        return nums[left];
    }
    
    int pivot_index = partition(nums, left, right);
    
    if (k == pivot_index) {
        return nums[k];
    } else if (k < pivot_index) {
        return quickselect(nums, left, pivot_index - 1, k);
    } else {
        return quickselect(nums, pivot_index + 1, right, k);
    }
}

int findKthLargest(vector<int>& nums, int k) {
    int size = nums.size();
    srand(time(0));
    return quickselect(nums, 0, size - 1, size - k);
}

int main() {
    int n, k;

    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> nums(n);

    cout << "Enter the elements:\n";
    for (int i = 0; i < n; ++i) {
        cin >> nums[i];
    }

    cout << "Enter the value of k: ";
    cin >> k;

    cout << findKthLargest(nums, k) << endl;  // Output the k-th largest element
    return 0;
}""",
            "python_code": """
import random

def partition(nums, left, right):
    pivot_index = random.randint(left, right)
    nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
    pivot = nums[right]
    i = left
    for j in range(left, right):
        if nums[j] <= pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[right] = nums[right], nums[i]
    return i

def quickselect(nums, left, right, k):
    if left == right:
        return nums[left]
    
    pivot_index = partition(nums, left, right)
    
    if k == pivot_index:
        return nums[k]
    elif k < pivot_index:
        return quickselect(nums, left, pivot_index - 1, k)
    else:
        return quickselect(nums, pivot_index + 1, right, k)

def findKthLargest(nums, k):
    size = len(nums)
    return quickselect(nums, 0, size - 1, size - k)

# Example usage
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(findKthLargest(nums, k))  # Output: 5
"""
        },
        {
            "title": "06 Graph Traversal Depth-First Search",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <unordered_map>
#include <stack>
#include <algorithm>

using namespace std;

void dfs(unordered_map<char, vector<char>>& graph, char start) {
    vector<char> visited;
    stack<char> stack;
    stack.push(start);

    while (!stack.empty()) {
        char node = stack.top();
        stack.pop();
        
        if (find(visited.begin(), visited.end(), node) == visited.end()) {
            visited.push_back(node);
            for (auto it = graph[node].rbegin(); it != graph[node].rend(); ++it) {
                stack.push(*it);
            }
        }
    }

    for (char node : visited) {
        cout << node << " ";
    }
    cout << endl;
}

int main() {
    unordered_map<char, vector<char>> graph;
    graph['A'] = {'B', 'C'};
    graph['B'] = {'D', 'E'};
    graph['C'] = {'F', 'G'};
    graph['D'] = {};
    graph['E'] = {};
    graph['F'] = {};
    graph['G'] = {};

    char start = 'A';
    dfs(graph, start);  // Output: A B D E C F G
    return 0;
}""",
            "python_code": """
def dfs(graph, start):
    visited = []
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            stack.extend(reversed(graph.get(node, [])))
    
    return visited

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}
start = 'A'
print(dfs(graph, start))  # Output: ['A', 'B', 'D', 'E', 'C', 'F', 'G']
"""
        },
        {
            "title": "07 Graph Traversal Breadth-First Search",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>

using namespace std;

void bfs(unordered_map<char, vector<char>>& graph, char start) {
    vector<char> visited;
    queue<char> queue;
    queue.push(start);

    while (!queue.empty()) {
        char node = queue.front();
        queue.pop();

        if (find(visited.begin(), visited.end(), node) == visited.end()) {
            visited.push_back(node);
            for (char neighbor : graph[node]) {
                queue.push(neighbor);
            }
        }
    }

    for (char node : visited) {
        cout << node << " ";
    }
    cout << endl;
}

int main() {
    unordered_map<char, vector<char>> graph;
    graph['A'] = {'B', 'C'};
    graph['B'] = {'D', 'E'};
    graph['C'] = {'F', 'G'};
    graph['D'] = {};
    graph['E'] = {};
    graph['F'] = {};
    graph['G'] = {};

    char start = 'A';
    bfs(graph, start);  // Output: A B C D E F G
    return 0;
}""",
            "python_code": """
def bfs(graph, start):
    visited = []
    queue = [start]
    
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(graph.get(node, []))
    
    return visited

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}
start = 'A'
print(bfs(graph, start))  # Output: ['A', 'B', 'C', 'D', 'E', 'F', 'G']
"""
        },
        {
            "title": "08 Dijkstra Algorithm",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>
#include <utility>

using namespace std;

unordered_map<char, int> dijkstra(unordered_map<char, unordered_map<char, int>>& graph, char start) {
    // Initialize distances with infinity
    unordered_map<char, int> distances;
    for (auto& node : graph) {
        distances[node.first] = numeric_limits<int>::max();
    }
    distances[start] = 0;
    
    // Priority queue to store the (distance, node)
    priority_queue<pair<int, char>, vector<pair<int, char>>, greater<pair<int, char>>> pq;
    pq.push({0, start});
    
    while (!pq.empty()) {
        int current_distance = pq.top().first;
        char current_node = pq.top().second;
        pq.pop();
        
        // If the popped node has a greater distance than the recorded one, continue
        if (current_distance > distances[current_node]) {
            continue;
        }
        
        // Update distances of neighbors
        for (auto& neighbor : graph[current_node]) {
            char neighbor_node = neighbor.first;
            int weight = neighbor.second;
            int distance = current_distance + weight;
            
            // If a shorter path is found
            if (distance < distances[neighbor_node]) {
                distances[neighbor_node] = distance;
                pq.push({distance, neighbor_node});
            }
        }
    }
    
    return distances;
}

int main() {
    unordered_map<char, unordered_map<char, int>> graph;
    graph['A'] = {{'B', 1}, {'C', 4}};
    graph['B'] = {{'C', 2}, {'D', 5}};
    graph['C'] = {{'D', 1}};
    graph['D'] = {};

    char start = 'A';
    unordered_map<char, int> result = dijkstra(graph, start);

    for (auto& pair : result) {
        cout << pair.first << ": " << pair.second << endl;
    }
    // Output: 
    // A: 0
    // B: 1
    // C: 3
    // D: 4

    return 0;
}""",
            "python_code": """
import heapq

def dijkstra(graph, start):
    # Initialize distances with infinity
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # Priority queue to store the (distance, node)
    pq = [(0, start)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        # If the popped node has a greater distance than the recorded one, continue
        if current_distance > distances[current_node]:
            continue
        
        # Update distances of neighbors
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            # If a shorter path is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {}
}
start = 'A'
print(dijkstra(graph, start))  # Output: {'A': 0, 'B': 1, 'C': 3, 'D': 4}
"""
        },
        {
            "title": "09 Median of Two Sorted Arrays",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    if (nums1.size() > nums2.size()) {
        return findMedianSortedArrays(nums2, nums1);
    }

    int x = nums1.size();
    int y = nums2.size();
    int low = 0, high = x;
    
    while (low <= high) {
        int partitionX = (low + high) / 2;
        int partitionY = (x + y + 1) / 2 - partitionX;
        
        int maxX = (partitionX == 0) ? INT_MIN : nums1[partitionX - 1];
        int minX = (partitionX == x) ? INT_MAX : nums1[partitionX];
        
        int maxY = (partitionY == 0) ? INT_MIN : nums2[partitionY - 1];
        int minY = (partitionY == y) ? INT_MAX : nums2[partitionY];
        
        if (maxX <= minY && maxY <= minX) {
            if ((x + y) % 2 == 0) {
                return ((double)max(maxX, maxY) + min(minX, minY)) / 2;
            } else {
                return (double)max(maxX, maxY);
            }
        } else if (maxX > minY) {
            high = partitionX - 1;
        } else {
            low = partitionX + 1;
        }
    }
    
    throw invalid_argument("Input arrays are not sorted");
}

int main() {
    int n1, n2;

    cout << "Enter the number of elements in the first array: ";
    cin >> n1;

    vector<int> nums1(n1);
    cout << "Enter the elements of the first array:\n";
    for (int i = 0; i < n1; ++i) {
        cin >> nums1[i];
    }

    cout << "Enter the number of elements in the second array: ";
    cin >> n2;

    vector<int> nums2(n2);
    cout << "Enter the elements of the second array:\n";
    for (int i = 0; i < n2; ++i) {
        cin >> nums2[i];
    }

    try {
        double median = findMedianSortedArrays(nums1, nums2);
        cout << "Median: " << median << endl;
    } catch (const invalid_argument& e) {
        cout << e.what() << endl;
    }

    return 0;
}""",
            "python_code": """
def findMedianSortedArrays(nums1, nums2):
    # Ensure nums1 is the smaller array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    x, y = len(nums1), len(nums2)
    low, high = 0, x
    
    while low <= high:
        partitionX = (low + high) // 2
        partitionY = (x + y + 1) // 2 - partitionX
        
        maxX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]
        minX = float('inf') if partitionX == x else nums1[partitionX]
        
        maxY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]
        minY = float('inf') if partitionY == y else nums2[partitionY]
        
        if maxX <= minY and maxY <= minX:
            if (x + y) % 2 == 0:
                return (max(maxX, maxY) + min(minX, minY)) / 2
            else:
                return max(maxX, maxY)
        elif maxX > minY:
            high = partitionX - 1
        else:
            low = partitionX + 1
    
    raise ValueError("Input arrays are not sorted")

# Example usage
nums1 = [1, 3]
nums2 = [2]
print(findMedianSortedArrays(nums1, nums2))  # Output: 2.0
"""
        },
        {
            "title": "10 N-Queens Problem",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <string>

using namespace std;

bool isSafe(vector<string> &board, int row, int col, int N) {
    for (int i = 0; i < row; i++) {
        if (board[i][col] == 'Q') return false;
        if (col - (row - i) >= 0 && board[i][col - (row - i)] == 'Q') return false;
        if (col + (row - i) < N && board[i][col + (row - i)] == 'Q') return false;
    }
    return true;
}

void solve(vector<vector<string>> &result, vector<string> &board, int row, int N) {
    if (row == N) {
        result.push_back(board);
        return;
    }
    for (int col = 0; col < N; col++) {
        if (isSafe(board, row, col, N)) {
            board[row][col] = 'Q';
            solve(result, board, row + 1, N);
            board[row][col] = '.';
        }
    }
}

vector<vector<string>> solveNQueens(int N) {
    vector<vector<string>> result;
    vector<string> board(N, string(N, '.'));
    solve(result, board, 0, N);
    return result;
}

int main() {
    int N = 4;
    vector<vector<string>> solutions = solveNQueens(N);
    for (const auto &solution : solutions) {
        for (const auto &row : solution) {
            cout << row << endl;
        }
        cout << endl;
    }
    // Output: [[".Q..", "...Q", "Q...", "..Q."], ["..Q.", "Q...", "...Q", ".Q.."]]
    return 0;
}""",
            "python_code": """
def solveNQueens(N):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False
            if col - (row - i) >= 0 and board[i][col - (row - i)] == 'Q':
                return False
            if col + (row - i) < N and board[i][col + (row - i)] == 'Q':
                return False
        return True
    
    def solve(board, row):
        if row == N:
            result.append([''.join(row) for row in board])
            return
        for col in range(N):
            if is_safe(board, row, col):
                board[row][col] = 'Q'
                solve(board, row + 1)
                board[row][col] = '.'
    
    result = []
    board = [['.' for _ in range(N)] for _ in range(N)]
    solve(board, 0)
    return result

# Example usage
N = 4
print(solveNQueens(N))
# Output: [['.Q..', '...Q', 'Q...', '..Q.'], ['..Q.', 'Q...', '...Q', '.Q..']]  
"""
        },
        {
            "title": "11 Trie Implementation",
            "cpp_code": """
#include <iostream>
#include <unordered_map>
using namespace std;

class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    bool is_end_of_word;

    TrieNode() : is_end_of_word(false) {}
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }
    
    void insert(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->is_end_of_word = true;
    }
    
    bool search(string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return node->is_end_of_word;
    }
    
    bool startsWith(string prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return true;
    }
};

int main() {
    Trie trie;
    trie.insert("hello");
    trie.insert("world");
    cout << trie.search("hello") << endl;  // Output: 1 (true)
    cout << trie.search("world") << endl;  // Output: 1 (true)
    cout << trie.search("hell") << endl;   // Output: 0 (false)
    cout << trie.startsWith("wor") << endl;  // Output: 1 (true)
    cout << trie.startsWith("worl") << endl; // Output: 1 (true)
    cout << trie.startsWith("worr") << endl; // Output: 0 (false)
    return 0;
}""",
            "python_code": """
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Example usage
trie = Trie()
trie.insert("hello")
trie.insert("world")
print(trie.search("hello"))  # Output: True
print(trie.search("world"))  # Output: True
print(trie.search("hell"))   # Output: False
print(trie.starts_with("wor"))  # Output: True
print(trie.starts_with("worl")) # Output: True
print(trie.starts_with("worr")) # Output: False
"""
        }
    ]
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(code_pairs, json_file, indent=4)

# Save the code pairs to a JSON file
save_code_pairs_to_json("Hand_code_pairs_Advance.json")
