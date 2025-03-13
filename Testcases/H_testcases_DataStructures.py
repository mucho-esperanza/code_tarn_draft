import json

def save_code_pairs_to_json(file_path):
    code_pairs = [
        {
            "title": "01 Binary Tree Inorder Traversal",
            "cpp_code": """
#include <iostream>     // include the iostream library for input and output
#include <vector>       // include the vector library for std::vector
#include <stack>        // include the stack library for std::stack

// Definition for a binary tree node.
struct TreeNode {
    int value;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : value(x), left(nullptr), right(nullptr) {}
};

std::vector<int> inorder_traversal(TreeNode* root) {
    std::vector<int> result;
    std::stack<TreeNode*> stack;
    TreeNode* current = root;
    
    while (current != nullptr || !stack.empty()) {
        while (current != nullptr) {
            stack.push(current);
            current = current->left;
        }
        
        current = stack.top();
        stack.pop();
        result.push_back(current->value);
        current = current->right;
    }
    
    return result;
}

// Helper function to build a tree from a list
TreeNode* build_tree_from_list(const std::vector<int>& values) {
    if (values.empty()) return nullptr;
    
    std::vector<TreeNode*> nodes;
    for (int val : values) {
        nodes.push_back(val == -1 ? nullptr : new TreeNode(val));
    }
    
    TreeNode* root = nodes[0];
    std::queue<TreeNode*> q;
    q.push(root);
    int i = 1;
    
    while (i < values.size()) {
        TreeNode* node = q.front();
        q.pop();
        
        if (node) {
            node->left = nodes[i++];
            if (node->left) q.push(node->left);
            
            if (i < values.size()) {
                node->right = nodes[i++];
                if (node->right) q.push(node->right);
            }
        }
    }
    
    return root;
}

int main() {
    std::vector<int> input_list = {1, -1, 2, 3}; // Use -1 for None
    TreeNode* root = build_tree_from_list(input_list);
    
    std::vector<int> result = inorder_traversal(root);
    std::cout << "Inorder traversal: ";
    for (int val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl; // Output: Inorder traversal: 1 3 2
    
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(root):
    result = []
    
    def traverse(node):
        if node:
            traverse(node.left)  # Traverse the left subtree
            result.append(node.value)  # Visit the root
            traverse(node.right)  # Traverse the right subtree
    
    traverse(root)
    return result

# Helper function to build a tree from a list
def build_tree_from_list(values):
    if not values:
        return None
    
    nodes = [None if val is None else TreeNode(val) for val in values]
    kids = nodes[::-1]
    root = kids.pop()
    
    for node in nodes:
        if node:
            if kids:
                node.left = kids.pop()
            if kids:
                node.right = kids.pop()
    
    return root

# Test the function with the input [1, None, 2, 3]
input_list = [1, None, 2, 3]
root = build_tree_from_list(input_list)
print(inorder_traversal(root))  # Output: [1, 3, 2]
"""
        },
        {
            "title": "02 Graph Cycle Detection",
            "cpp_code": """
#include <iostream>            // include the iostream library for input and output
#include <unordered_map>       // include the unordered_map library for std::unordered_map
#include <unordered_set>       // include the unordered_set library for std::unordered_set
#include <vector>              // include the vector library for std::vector

using namespace std;

// Helper function for DFS
bool dfs(const string& node, unordered_map<string, vector<string>>& graph, 
         unordered_set<string>& visited, unordered_set<string>& recursion_stack) {
    visited.insert(node);
    recursion_stack.insert(node);
    
    for (const string& neighbor : graph[node]) {
        if (recursion_stack.find(neighbor) != recursion_stack.end()) {
            return true;  // Found a cycle
        }
        if (visited.find(neighbor) == visited.end()) {
            if (dfs(neighbor, graph, visited, recursion_stack)) {
                return true;
            }
        }
    }
    
    recursion_stack.erase(node);
    return false;
}

// Function to check if the graph contains a cycle
bool has_cycle(unordered_map<string, vector<string>>& graph) {
    unordered_set<string> visited;
    unordered_set<string> recursion_stack;
    
    for (const auto& pair : graph) {
        if (visited.find(pair.first) == visited.end()) {
            if (dfs(pair.first, graph, visited, recursion_stack)) {
                return true;
            }
        }
    }
    return false;
}

int main() {
    unordered_map<string, vector<string>> graph = {
        {"A", {"B"}},
        {"B", {"C"}},
        {"C", {"A"}}
    };
    
    cout << (has_cycle(graph) ? "True" : "False") << endl;  // Output: True
    
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
def has_cycle(graph):
    def dfs(node, visited, recursion_stack):
        visited.add(node)
        recursion_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor, visited, recursion_stack):
                    return True
            elif neighbor in recursion_stack:
                return True
        
        recursion_stack.remove(node)
        return False

    visited = set()
    recursion_stack = set()
    
    for node in graph:
        if node not in visited:
            if dfs(node, visited, recursion_stack):
                return True
    return False

# Test the function with the input graph
graph = {'A': ['B'], 'B': ['C'], 'C': ['A']}
print(has_cycle(graph))  # Output: True
"""
        },
        {
            "title": "03 Max Heap Implementation",
            "cpp_code": """
#include <iostream>        // include the iostream library for input and output
#include <vector>          // include the vector library for std::vector>
#include <stdexcept>       // include the stdexcept library for std::out_of_range exception

class MaxHeap {
public:
    MaxHeap() {}

    void insert(int value) {
        heap.push_back(value);
        heapify_up(heap.size() - 1);
    }

    int extract_max() {
        if (heap.empty()) {
            throw std::out_of_range("Heap is empty");
        }
        int max_value = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        if (!heap.empty()) {
            heapify_down(0);
        }
        return max_value;
    }

private:
    std::vector<int> heap;

    void heapify_up(int index) {
        int parent_index = (index - 1) / 2;
        if (index > 0 && heap[index] > heap[parent_index]) {
            std::swap(heap[index], heap[parent_index]);
            heapify_up(parent_index);
        }
    }

    void heapify_down(int index) {
        int left_child_index = 2 * index + 1;
        int right_child_index = 2 * index + 2;
        int largest = index;
        
        if (left_child_index < heap.size() && heap[left_child_index] > heap[largest]) {
            largest = left_child_index;
        }
        
        if (right_child_index < heap.size() && heap[right_child_index] > heap[largest]) {
            largest = right_child_index;
        }
        
        if (largest != index) {
            std::swap(heap[index], heap[largest]);
            heapify_down(largest);
        }
    }
};

int main() {
    MaxHeap max_heap;
    max_heap.insert(3);
    max_heap.insert(2);
    max_heap.insert(15);
    std::cout << max_heap.extract_max() << std::endl;  // Output: 15
    
    return 0;  // return 0 to indicate the program ended successfully
}""",
            "python_code": """
class MaxHeap:
    def __init__(self):
        self.heap = []

    def insert(self, value):
        # Add the new value at the end of the heap
        self.heap.append(value)
        self._heapify_up(len(self.heap) - 1)

    def extract_max(self):
        if not self.heap:
            raise IndexError("Heap is empty")
        # The max value is at the root
        max_value = self.heap[0]
        # Move the last value to the root and heapify down
        self.heap[0] = self.heap.pop()
        if self.heap:
            self._heapify_down(0)
        return max_value

    def _heapify_up(self, index):
        # Move the element at index up to maintain the heap property
        parent_index = (index - 1) // 2
        if index > 0 and self.heap[index] > self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            self._heapify_up(parent_index)

    def _heapify_down(self, index):
        # Move the element at index down to maintain the heap property
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        largest = index
        
        if (left_child_index < len(self.heap) and
            self.heap[left_child_index] > self.heap[largest]):
            largest = left_child_index
        
        if (right_child_index < len(self.heap) and
            self.heap[right_child_index] > self.heap[largest]):
            largest = right_child_index
        
        if largest != index:
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            self._heapify_down(largest)

# Test the MaxHeap
max_heap = MaxHeap()
max_heap.insert(3)
max_heap.insert(2)
max_heap.insert(15)
print(max_heap.extract_max())  # Output: 15
"""
        },
        {
            "title": "04 Inorder Successor in BST",
            "cpp_code": """
#include <iostream>            // include the iostream library for input and output
#include <vector>              // include the vector library for std::vector>

// Definition for a binary tree node.
struct TreeNode {
    int key;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : key(x), left(nullptr), right(nullptr) {}
};

// Helper function to find the minimum value node in a subtree
TreeNode* min_value_node(TreeNode* node) {
    TreeNode* current = node;
    while (current && current->left != nullptr) {
        current = current->left;
    }
    return current;
}

// Function to find the inorder successor of a given node
TreeNode* find_inorder_successor(TreeNode* root, TreeNode* node) {
    if (node->right != nullptr) {
        return min_value_node(node->right);
    }
    
    TreeNode* successor = nullptr;
    TreeNode* ancestor = root;
    
    while (ancestor != node) {
        if (node->key < ancestor->key) {
            successor = ancestor;
            ancestor = ancestor->left;
        } else {
            ancestor = ancestor->right;
        }
    }
    
    return successor;
}

// Helper function to build a tree from a list
TreeNode* build_tree_from_list(const std::vector<int>& values, int index = 0) {
    if (index >= values.size() || values[index] == -1) {
        return nullptr;
    }
    
    TreeNode* node = new TreeNode(values[index]);
    node->left = build_tree_from_list(values, 2 * index + 1);
    node->right = build_tree_from_list(values, 2 * index + 2);
    
    return node;
}

int main() {
    std::vector<int> input_list = {20, 8, 22, 4, 12, -1, -1, -1, -1, 10, 14}; // Use -1 for None
    TreeNode* root = build_tree_from_list(input_list);
    TreeNode* node = root; // Assuming the node with key 8 is the target
    while (node && node->key != 8) {
        if (8 < node->key) {
            node = node->left;
        } else {
            node = node->right;
        }
    }
    
    TreeNode* successor = find_inorder_successor(root, node);
    std::cout << (successor ? successor->key : -1) << std::endl;  // Output: 10
    
    return 0;  // return 0 to indicate the program ended successfully
}""",
            "python_code": """
class TreeNode:
    def __init__(self, key, left=None, right=None):
        self.key = key
        self.left = left
        self.right = right

def inorder_successor(root, n):
    successor = None
    
    while root:
        if n.key < root.key:
            successor = root
            root = root.left
        else:
            root = root.right
    
    return successor

def min_value_node(node):
    current = node
    while current.left:
        current = current.left
    return current

def find_inorder_successor(root, key):
    # Find the node
    node = root
    while node and node.key != key:
        if key < node.key:
            node = node.left
        else:
            node = node.right
    
    if not node:
        return None

    if node.right:
        return min_value_node(node.right)
    
    successor = None
    ancestor = root
    while ancestor != node:
        if node.key < ancestor.key:
            successor = ancestor
            ancestor = ancestor.left
        else:
            ancestor = ancestor.right
    
    return successor

# Helper function to build a tree from a list
def build_tree_from_list(values):
    if not values:
        return None
    
    nodes = [None if val is None else TreeNode(val) for val in values]
    kids = nodes[::-1]
    root = kids.pop()
    
    for node in nodes:
        if node:
            if kids:
                node.left = kids.pop()
            if kids:
                node.right = kids.pop()
    
    return root

# Test the function with the input BST
input_list = [20, 8, 22, 4, 12, None, None, None, None, 10, 14]
root = build_tree_from_list(input_list)
node_key = 8
successor = find_inorder_successor(root, node_key)
print(successor.key if successor else "None")  # Output: 10
"""
        },
        {
            "title": "05 Find All Paths in a Graph",
            "cpp_code": """
#include <iostream>        // for input and output
#include <vector>          // for std::vector
#include <unordered_map>   // for std::unordered_map
#include <unordered_set>   // for std::unordered_set

void find_all_paths_dfs(const std::unordered_map<char, std::vector<char>>& graph, char current, char end, std::vector<char>& path, std::vector<std::vector<char>>& paths) {
    path.push_back(current);
    
    if (current == end) {
        paths.push_back(path);
    } else {
        for (char neighbor : graph.at(current)) {
            if (std::find(path.begin(), path.end(), neighbor) == path.end()) {  // Avoid cycles
                find_all_paths_dfs(graph, neighbor, end, path, paths);
            }
        }
    }
    
    path.pop_back();  // Backtrack
}

std::vector<std::vector<char>> find_all_paths(const std::unordered_map<char, std::vector<char>>& graph, char start, char end) {
    std::vector<std::vector<char>> paths;
    std::vector<char> path;
    find_all_paths_dfs(graph, start, end, path, paths);
    return paths;
}

int main() {
    std::unordered_map<char, std::vector<char>> graph = {
        {'A', {'B', 'C'}},
        {'B', {'C', 'D'}},
        {'C', {'D'}},
        {'D', {}}
    };
    char start = 'A';
    char end = 'D';
    
    std::vector<std::vector<char>> all_paths = find_all_paths(graph, start, end);
    for (const auto& path : all_paths) {
        for (char node : path) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;  // Return 0 to indicate the program ended successfully
}""",
            "python_code": """
def find_all_paths(graph, start, end):
    def dfs(current_node, path):
        # Add the current node to the path
        path.append(current_node)
        
        # If the current node is the end node, add the path to results
        if current_node == end:
            paths.append(path.copy())
        else:
            # Explore all neighbors
            for neighbor in graph.get(current_node, []):
                if neighbor not in path:  # Avoid cycles
                    dfs(neighbor, path)
        
        # Backtrack: Remove the current node from path
        path.pop()

    paths = []
    dfs(start, [])
    return paths

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['D'],
    'D': []
}
start = 'A'
end = 'D'
all_paths = find_all_paths(graph, start, end)
print(all_paths)  # Output: [['A', 'B', 'D'], ['A', 'B', 'C', 'D'], ['A', 'C', 'D']]
"""
        },
        {
            "title": "06 Word Search in Grid",
            "cpp_code": """
#include <iostream>              // for input and output
#include <vector>                // for std::vector

bool dfs(const std::vector<std::vector<char>>& board, std::vector<std::vector<bool>>& visited, const std::string& word, int i, int j, int k) {
    if (k == word.size()) return true;
    if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size() || board[i][j] != word[k] || visited[i][j]) return false;

    visited[i][j] = true; // Mark the cell as visited

    // Explore all 4 directions
    bool found = dfs(board, visited, word, i + 1, j, k + 1) ||
                 dfs(board, visited, word, i - 1, j, k + 1) ||
                 dfs(board, visited, word, i, j + 1, k + 1) ||
                 dfs(board, visited, word, i, j - 1, k + 1);

    visited[i][j] = false; // Unmark the cell
    return found;
}

bool exist(const std::vector<std::vector<char>>& board, const std::string& word) {
    if (board.empty() || board[0].empty()) return false;

    std::vector<std::vector<bool>> visited(board.size(), std::vector<bool>(board[0].size(), false));

    for (int i = 0; i < board.size(); ++i) {
        for (int j = 0; j < board[0].size(); ++j) {
            if (dfs(board, visited, word, i, j, 0)) return true;
        }
    }
    
    return false;
}

int main() {
    std::vector<std::vector<char>> board = {
        {'A', 'B', 'C', 'E'},
        {'S', 'F', 'C', 'S'},
        {'A', 'D', 'E', 'E'}
    };
    std::string word = "ABCCED";
    
    std::cout << (exist(board, word) ? "True" : "False") << std::endl; // Output: True
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
def exist(board, word):
    def dfs(board, word, i, j, k):
        if k == len(word):
            return True
        if (i < 0 or i >= len(board) or j < 0 or j >= len(board) or
            board[i][j] != word[k]):
            return False
        
        # Save the current cell's value and mark it as visited
        temp = board[i][j]
        board[i][j] = '#'
        
        # Explore all 4 directions: up, down, left, right
        found = (dfs(board, word, i + 1, j, k + 1) or
                 dfs(board, word, i - 1, j, k + 1) or
                 dfs(board, word, i, j + 1, k + 1) or
                 dfs(board, word, i, j - 1, k + 1))
        
        # Restore the current cell's value
        board[i][j] = temp
        
        return found

    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(board, word, i, j, 0):
                return True
    
    return False

# Example usage
board = [
    ["A", "B", "C", "E"],
    ["S", "F", "C", "S"],
    ["A", "D", "E", "E"]
]
word = "ABCCED"
print(exist(board, word))  # Output: True
"""
        },
        {
            "title": "07 Implement Stack Using Queues by Make Push Operation Costly",
            "cpp_code": """
#include <iostream>              // for input and output
#include <queue>                 // for std::queue

class StackUsingQueues {
public:
    void push(int x) {
        queue2.push(x);
        while (!queue1.empty()) {
            queue2.push(queue1.front());
            queue1.pop();
        }
        std::swap(queue1, queue2);
    }

    int pop() {
        if (queue1.empty()) throw std::out_of_range("Pop from an empty stack");
        int value = queue1.front();
        queue1.pop();
        return value;
    }

    int top() {
        if (queue1.empty()) throw std::out_of_range("Top from an empty stack");
        return queue1.front();
    }

    bool empty() {
        return queue1.empty();
    }

private:
    std::queue<int> queue1;
    std::queue<int> queue2;
};

// Example usage
int main() {
    StackUsingQueues stack;
    stack.push(1);
    stack.push(2);
    std::cout << stack.pop() << std::endl;  // Output: 2
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
from collections import deque

class StackUsingQueues:
    def __init__(self):
        self.queue1 = deque()
        self.queue2 = deque()

    def push(self, x):
        self.queue2.append(x)
        while self.queue1:
            self.queue2.append(self.queue1.popleft())
        self.queue1, self.queue2 = self.queue2, self.queue1

    def pop(self):
        if not self.queue1:
            raise IndexError("Pop from an empty stack")
        return self.queue1.popleft()

    def top(self):
        if not self.queue1:
            raise IndexError("Top from an empty stack")
        return self.queue1[0]

    def empty(self):
        return not self.queue1

# Example usage
stack = StackUsingQueues()
stack.push(1)
stack.push(2)
print(stack.pop())  # Output: 2
"""
        },
        {
            "title": "08 Implement Stack Using Queues by Make Pop Operation Costly",
            "cpp_code": """
#include <iostream>              // for input and output
#include <queue>                 // for std::queue

class StackUsingQueues {
public:
    void push(int x) {
        queue1.push(x);
    }

    int pop() {
        if (queue1.empty()) throw std::out_of_range("Pop from an empty stack");
        while (queue1.size() > 1) {
            queue2.push(queue1.front());
            queue1.pop();
        }
        int value = queue1.front();
        queue1.pop();
        std::swap(queue1, queue2);
        return value;
    }

    int top() {
        if (queue1.empty()) throw std::out_of_range("Top from an empty stack");
        while (queue1.size() > 1) {
            queue2.push(queue1.front());
            queue1.pop();
        }
        int value = queue1.front();
        queue2.push(value);
        queue1.pop();
        std::swap(queue1, queue2);
        return value;
    }

    bool empty() {
        return queue1.empty();
    }

private:
    std::queue<int> queue1;
    std::queue<int> queue2;
};

// Example usage
int main() {
    StackUsingQueues stack;
    stack.push(1);
    stack.push(2);
    std::cout << stack.pop() << std::endl;  // Output: 2
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
from collections import deque

class StackUsingQueues:
    def __init__(self):
        self.queue1 = deque()
        self.queue2 = deque()

    def push(self, x):
        self.queue1.append(x)

    def pop(self):
        if not self.queue1:
            raise IndexError("Pop from an empty stack")
        while len(self.queue1) > 1:
            self.queue2.append(self.queue1.popleft())
        value = self.queue1.popleft()
        self.queue1, self.queue2 = self.queue2, self.queue1
        return value

    def top(self):
        if not self.queue1:
            raise IndexError("Top from an empty stack")
        while len(self.queue1) > 1:
            self.queue2.append(self.queue1.popleft())
        value = self.queue1[0]
        self.queue2.append(self.queue1.popleft())
        self.queue1, self.queue2 = self.queue2, self.queue1
        return value

    def empty(self):
        return not self.queue1

# Example usage
stack = StackUsingQueues()
stack.push(1)
stack.push(2)
print(stack.pop())  # Output: 2
"""
        },
        {
            "title": "09 Sort a Stack",
            "cpp_code": """
#include <iostream>              // for input and output
#include <stack>                 // for std::stack

void sortStack(std::stack<int>& stack) {
    std::stack<int> auxStack;
    
    while (!stack.empty()) {
        int temp = stack.top();
        stack.pop();
        
        while (!auxStack.empty() && auxStack.top() > temp) {
            stack.push(auxStack.top());
            auxStack.pop();
        }
        
        auxStack.push(temp);
    }
    
    while (!auxStack.empty()) {
        stack.push(auxStack.top());
        auxStack.pop();
    }
}

// Example usage
int main() {
    std::stack<int> stack;
    stack.push(34);
    stack.push(3);
    stack.push(31);
    stack.push(98);
    stack.push(92);
    stack.push(23);

    sortStack(stack);

    // Print sorted stack
    while (!stack.empty()) {
        std::cout << stack.top() << " ";
        stack.pop();
    }
    std::cout << std::endl;  // Output: 3 23 31 34 92 98
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
def sort_stack(stack):
    # Create an auxiliary stack
    aux_stack = []

    while stack:
        # Pop the top element from the main stack
        temp = stack.pop()
        
        # While the auxiliary stack is not empty and the top of the auxiliary stack is greater than temp
        while aux_stack and aux_stack[-1] > temp:
            stack.append(aux_stack.pop())
        
        # Push the temp element onto the auxiliary stack
        aux_stack.append(temp)
    
    # Transfer the sorted elements back to the main stack
    while aux_stack:
        stack.append(aux_stack.pop())

# Example usage
stack = [34, 3, 31, 98, 92, 23]
sort_stack(stack)
print(stack)  # Output: [3, 23, 31, 34, 92, 98]
"""
        },
        {
            "title": "10 Circular Queue Implementation",
            "cpp_code": """
#include <iostream>              // for input and output
#include <vector>                // for std::vector

class CircularQueue {
public:
    CircularQueue(int size) : size(size), front(0), rear(-1), count(0) {
        queue.resize(size);
    }

    void enqueue(int value) {
        if (isFull()) throw std::out_of_range("Queue is full");
        rear = (rear + 1) % size;
        queue[rear] = value;
        ++count;
    }

    int dequeue() {
        if (isEmpty()) throw std::out_of_range("Queue is empty");
        int value = queue[front];
        front = (front + 1) % size;
        --count;
        return value;
    }

    int frontElement() {
        if (isEmpty()) throw std::out_of_range("Queue is empty");
        return queue[front];
    }

    bool isEmpty() {
        return count == 0;
    }

    bool isFull() {
        return count == size;
    }

    void printQueue() {
        for (int i = 0; i < count; ++i) {
            std::cout << queue[(front + i) % size] << " ";
        }
        std::cout << std::endl;
    }

private:
    int size;
    int front;
    int rear;
    int count;
    std::vector<int> queue;
};

// Example usage
int main() {
    CircularQueue cq(5);
    cq.enqueue(1);
    cq.enqueue(2);
    cq.dequeue();
    cq.enqueue(3);
    cq.printQueue();  // Output: 2 3
    
    return 0;  // Return 0 to indicate successful completion
}""",
            "python_code": """
class CircularQueue:
    def __init__(self, size):
        self.size = size
        self.queue = [None] * size
        self.front = 0
        self.rear = -1
        self.count = 0

    def enqueue(self, value):
        if self.is_full():
            raise Exception("Queue is full")
        self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = value
        self.count += 1

    def dequeue(self):
        if self.is_empty():
            raise Exception("Queue is empty")
        value = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.size
        self.count -= 1
        return value

    def front_element(self):
        if self.is_empty():
            raise Exception("Queue is empty")
        return self.queue[self.front]

    def is_empty(self):
        return self.count == 0

    def is_full(self):
        return self.count == self.size

    def __str__(self):
        return str([self.queue[(self.front + i) % self.size] for i in range(self.count)])

# Example usage
cq = CircularQueue(5)
cq.enqueue(1)
cq.enqueue(2)
cq.dequeue()
cq.enqueue(3)
print(cq)  # Output: [2, 3]
"""
        },
        {
            "title": "11 String manipulation",
            "cpp_code": """
string reverseString(string str) {
    int n = str.length();
    for(int i = 0; i < n/2; i++) {
        swap(str[i], str[n-1-i]);
    }
    return str;
}""",
            "python_code": """
def reverse_string(string):
    return string[::-1]
"""
        }
    ]
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(code_pairs, json_file, indent=4)

# Save the code pairs to a JSON file
save_code_pairs_to_json("Hand_code_pairs_DataStructure.json")
