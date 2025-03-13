import json

def save_code_pairs_to_json(file_path):
    code_pairs = [
        {
            "title": "01 Binary Search",
            "cpp_code": """
#include <iostream> // include the iostream library for input and output
#include <vector>   // include the vector library for std::vector

int binary_search(const std::vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2; // Avoid overflow
        if (arr[mid] == target) {
            return mid; // Return the index if the target is found
        } else if (arr[mid] < target) {
            left = mid + 1; // Search in the right half
        } else {
            right = mid - 1; // Search in the left half
        }
    }
    
    return -1; // Return -1 if the target is not found
}

int main() {
    std::vector<int> input_array = {1, 2, 3, 4, 5, 6};
    int target_value = 4;
    int index = binary_search(input_array, target_value);
    std::cout << "Index: " << index << std::endl; // Output: Index: 3
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid  # Return the index if the target is found
        elif arr[mid] < target:
            left = mid + 1  # Search in the right half
        else:
            right = mid - 1  # Search in the left half
            
    return -1  # Return -1 if the target is not found

# Test the function with the input [1, 2, 3, 4, 5, 6] and target 4
input_array = [1, 2, 3, 4, 5, 6]
target_value = 4
index = binary_search(input_array, target_value)
print(f"Index: {index}") # Output: Index: 3
"""
        },
        {
            "title": "02 Anagram Check",
            "cpp_code": """
#include <iostream> // include the iostream library for input and output
#include <algorithm> // include the algorithm library for std::sort
#include <cctype> // include the cctype library for std::tolower
#include <string> // include the string library for std::string

bool are_anagrams(const std::string& str1, const std::string& str2) {
    // Convert both strings to lowercase
    std::string s1 = str1;
    std::string s2 = str2;
    for (char& c : s1) {
        c = std::tolower(c);
    }
    for (char& c : s2) {
        c = std::tolower(c);
    }
    
    // Remove spaces from both strings
    s1.erase(std::remove(s1.begin(), s1.end(), ' '), s1.end());
    s2.erase(std::remove(s2.begin(), s2.end(), ' '), s2.end());
    
    // Check if lengths are the same
    if (s1.length() != s2.length()) {
        return false;
    }
    
    // Sort both strings and compare
    std::sort(s1.begin(), s1.end());
    std::sort(s2.begin(), s2.end());
    return s1 == s2;
}

int main() {
    std::string input_str1 = "listen";
    std::string input_str2 = "silent";
    std::cout << std::boolalpha << are_anagrams(input_str1, input_str2) << std::endl; // Output: true
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
def are_anagrams(str1, str2):
    # Remove spaces and convert to lowercase
    str1 = str1.replace(" ", "").lower()
    str2 = str2.replace(" ", "").lower()
    
    # Check if lengths are the same
    if len(str1) != len(str2):
        return False
    
    # Sort both strings and compare
    return sorted(str1) == sorted(str2)

# Test the function with the input "listen" and "silent"
input_str1 = "listen"
input_str2 = "silent"
print(are_anagrams(input_str1, input_str2)) # Output: True
"""
        },
        {
            "title": "03 Merge Sort",
            "cpp_code": """
#include <iostream> // include the iostream library for input and output
#include <vector>   // include the vector library for std::vector

// Merge function to merge two halves
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Create temporary vectors
    std::vector<int> L(n1);
    std::vector<int> R(n2);

    // Copy data to temp vectors L[] and R[]
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    // Merge the temp vectors back into arr[left..right]
    int i = 0;
    int j = 0;
    int k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Function to sort the array using merge sort
void merge_sort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Sort first and second halves
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);

        // Merge the sorted halves
        merge(arr, left, mid, right);
    }
}

int main() {
    std::vector<int> input_array = {4, 2, 5, 1, 3};
    merge_sort(input_array, 0, input_array.size() - 1);
    
    for (int num : input_array) {
        std::cout << num << " "; // Print the sorted array
    }
    std::cout << std::endl;
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Finding the mid of the array
        left_half = arr[:mid]  # Dividing the elements into 2 halves
        right_half = arr[mid:]

        merge_sort(left_half)  # Sorting the first half
        merge_sort(right_half)  # Sorting the second half

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

# Test the function with the input [4, 2, 5, 1, 3]
input_array = [4, 2, 5, 1, 3]
merge_sort(input_array)
print(input_array) # Output: [1, 2, 3, 4, 5]
"""
        },
        {
            "title": "04 Linked List Implementation",
            "cpp_code": """
#include <iostream> // include the iostream library for input and output

// Define the Node structure
struct Node {
    int data;
    Node* next;
    
    Node(int data) : data(data), next(nullptr) {}
};

// Define the LinkedList class
class LinkedList {
public:
    LinkedList() : head(nullptr) {}
    
    // Add a node with the given data to the front of the list
    void add(int data) {
        Node* new_node = new Node(data);
        new_node->next = head;
        head = new_node;
    }
    
    // Remove the first node with the given data
    void remove(int data) {
        Node* current = head;
        Node* previous = nullptr;
        
        while (current != nullptr) {
            if (current->data == data) {
                if (previous == nullptr) {
                    head = current->next;
                } else {
                    previous->next = current->next;
                }
                delete current;
                return;
            }
            previous = current;
            current = current->next;
        }
    }
    
    // Print the linked list
    void print() const {
        Node* current = head;
        std::cout << "[";
        while (current != nullptr) {
            std::cout << current->data;
            if (current->next != nullptr) {
                std::cout << ", ";
            }
            current = current->next;
        }
        std::cout << "]" << std::endl;
    }
    
    ~LinkedList() {
        Node* current = head;
        while (current != nullptr) {
            Node* next = current->next;
            delete current;
            current = next;
        }
    }
    
private:
    Node* head;
};

int main() {
    LinkedList linked_list;
    linked_list.add(1);
    linked_list.add(2);
    linked_list.remove(1);
    linked_list.print(); // Output: [2]
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def add(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def remove(self, data):
        current = self.head
        previous = None
        
        while current is not None:
            if current.data == data:
                if previous is None:
                    self.head = current.next
                else:
                    previous.next = current.next
                return
            previous = current
            current = current.next
    
    def __str__(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return str(elements)

# Create a linked list and manipulate it according to the input
linked_list = LinkedList()
linked_list.add(1)
linked_list.add(2)
linked_list.remove(1)
print(linked_list) # Output: [2]
"""
        },
        {
            "title": "05 Queue Using Two Stacks",
            "cpp_code": """
#include <iostream> // include the iostream library for input and output
#include <stack>    // include the stack library for std::stack

class QueueUsingTwoStacks {
public:
    void enqueue(int item) {
        // Push item onto stack1
        stack1.push(item);
    }
    
    int dequeue() {
        if (stack2.empty()) {
            // Transfer elements from stack1 to stack2, if stack2 is empty
            while (!stack1.empty()) {
                stack2.push(stack1.top());
                stack1.pop();
            }
        }
        if (!stack2.empty()) {
            int item = stack2.top();
            stack2.pop();
            return item;
        } else {
            throw std::out_of_range("Dequeue from empty queue");
        }
    }
    
private:
    std::stack<int> stack1;
    std::stack<int> stack2;
};

int main() {
    QueueUsingTwoStacks queue;
    queue.enqueue(1);
    queue.enqueue(2);
    try {
        int dequeued_item = queue.dequeue();
        std::cout << "Dequeued: " << dequeued_item << std::endl; // Output: Dequeued: 1
    } catch (const std::out_of_range& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
class QueueUsingTwoStacks:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def enqueue(self, item):
        # Push item onto stack1
        self.stack1.append(item)
    
    def dequeue(self):
        if not self.stack2:
            # Transfer elements from stack1 to stack2, if stack2 is empty
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        if self.stack2:
            return self.stack2.pop()
        else:
            raise IndexError("Dequeue from empty queue")

# Create a queue and manipulate it according to the input
queue = QueueUsingTwoStacks()
queue.enqueue(1)
queue.enqueue(2)
dequeued_item = queue.dequeue()
print(f"Dequeued: {dequeued_item}") # Output: Dequeued: 1
"""
        },
        {
            "title": "06 Balanced Parentheses",
            "cpp_code": """
#include <iostream> // include the iostream library for input and output
#include <stack>    // include the stack library for std::stack
#include <unordered_map> // include the unordered_map library for std::unordered_map

bool is_balanced(const std::string& s) {
    std::stack<char> stack;
    std::unordered_map<char, char> matching_parenthesis = {
        {')', '('},
        {'}', '{'},
        {']', '['}
    };
    
    for (char char : s) {
        if (matching_parenthesis.count(char)) {
            // If it's a closing bracket, check for matching opening bracket
            if (stack.empty() || stack.top() != matching_parenthesis[char]) {
                return false;
            }
            stack.pop();
        } else if (char == '(' || char == '{' || char == '[') {
            // If it's an opening bracket, push it onto the stack
            stack.push(char);
        } else {
            // If it's not a parenthesis, ignore
            continue;
        }
    }
    
    // If stack is empty, all parentheses were matched
    return stack.empty();
}

int main() {
    std::string input_string = "((()))";
    std::cout << (is_balanced(input_string) ? "True" : "False") << std::endl; // Output: True
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
def is_balanced(s):
    stack = []
    # Dictionary to hold matching pairs
    matching_parenthesis = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in matching_parenthesis.values():
            # If it's an opening bracket, push it onto the stack
            stack.append(char)
        elif char in matching_parenthesis.keys():
            # If it's a closing bracket, check for matching opening bracket
            if stack == [] or matching_parenthesis[char] != stack.pop():
                return False
        else:
            # If it's not a parenthesis, ignore
            continue
    
    # If stack is empty, all parentheses were matched
    return stack == []

# Test the function with the input "((()))"
input_string = "((()))"
print(is_balanced(input_string))  # Output: True
"""
        },
        {
            "title": "07 Longest Substring Without Repeating Characters",
            "cpp_code": """
#include <iostream>       // include the iostream library for input and output
#include <unordered_map> // include the unordered_map library for std::unordered_map
#include <string>         // include the string library for std::string

std::string longest_substring_without_repeating(const std::string& s) {
    std::unordered_map<char, int> char_index; // Dictionary to store the last index of each character
    int start = 0; // Start index of the current substring
    int max_length = 0; // Maximum length of substring without repeating characters
    std::string longest_substr; // The longest substring without repeating characters
    
    for (int end = 0; end < s.length(); ++end) {
        char char_end = s[end];
        
        if (char_index.count(char_end) && char_index[char_end] >= start) {
            // Move the start index to one past the last occurrence of the current character
            start = char_index[char_end] + 1;
        }
        
        char_index[char_end] = end; // Update the last index of the current character
        
        int current_length = end - start + 1; // Calculate current substring length
        
        if (current_length > max_length) {
            max_length = current_length;
            longest_substr = s.substr(start, current_length);
        }
    }
    
    return longest_substr;
}

int main() {
    std::string input_string = "abcabcbb";
    std::cout << longest_substring_without_repeating(input_string) << std::endl; // Output: "abc"
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
def longest_substring_without_repeating(s):
    char_index = {}  # Dictionary to store the last index of each character
    start = 0  # Start index of the current substring
    max_length = 0  # Maximum length of substring without repeating characters
    longest_substr = ""  # The longest substring without repeating characters
    
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            # Move the start index to one past the last occurrence of the current character
            start = char_index[char] + 1
        
        char_index[char] = end  # Update the last index of the current character
        current_length = end - start + 1  # Calculate current substring length
        
        if current_length > max_length:
            max_length = current_length
            longest_substr = s[start:end + 1]
    
    return longest_substr

# Test the function with the input "abcabcbb"
input_string = "abcabcbb"
print(longest_substring_without_repeating(input_string))  # Output: "abc"
"""
        },
        {
            "title": "08 Sum of Digits",
            "cpp_code": """
#include <iostream> // include the iostream library for input and output

int sum_of_digits(int n) {
    // This function calculates the sum of digits of a number until a single digit is obtained
    while (n >= 10) { // Continue until n is a single digit
        int sum = 0;
        while (n > 0) {
            sum += n % 10; // Add the last digit to sum
            n /= 10;       // Remove the last digit from n
        }
        n = sum; // Update n to the new sum
    }
    return n;
}

int main() {
    int input_number = 38;
    std::cout << sum_of_digits(input_number) << std::endl; // Output: 2
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
def sum_of_digits(n):
    # This function calculates the sum of digits of a number until a single digit is obtained
    while n >= 10:  # Continue until n is a single digit
        n = sum(int(digit) for digit in str(n))  # Sum of digits
    return n

# Test the function with the input 38
input_number = 38
print(sum_of_digits(input_number))  # Output: 2
"""
        },
        {
            "title": "09 Find Duplicates in Array",
            "cpp_code": """
#include <iostream>       // include the iostream library for input and output
#include <unordered_set> // include the unordered_set library for std::unordered_set
#include <vector>         // include the vector library for std::vector

std::vector<int> find_duplicates(const std::vector<int>& arr) {
    std::unordered_set<int> seen;      // Set to keep track of seen elements
    std::unordered_set<int> duplicates; // Set to keep track of duplicates

    for (int num : arr) {
        if (seen.find(num) != seen.end()) {
            // If the number is already in 'seen', it's a duplicate
            duplicates.insert(num);
        } else {
            // Otherwise, add it to 'seen'
            seen.insert(num);
        }
    }

    // Convert duplicates set to vector and return
    return std::vector<int>(duplicates.begin(), duplicates.end());
}

int main() {
    std::vector<int> input_array = {1, 2, 3, 1, 4, 2};
    std::vector<int> result = find_duplicates(input_array);
    
    std::cout << "Duplicates: ";
    for (int num : result) {
        std::cout << num << " ";
    }
    std::cout << std::endl; // Output: Duplicates: 1 2
    
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
def find_duplicates(arr):
    seen = set()
    duplicates = set()
    
    for num in arr:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    
    return list(duplicates)

# Test the function with the input [1, 2, 3, 1, 4, 2]
input_array = [1, 2, 3, 1, 4, 2]
print(find_duplicates(input_array))  # Output: [1, 2]
"""
        },
        {
            "title": "10 Rotate Matrix",
            "cpp_code": """
#include <iostream>         // include the iostream library for input and output
#include <vector>           // include the vector library for std::vector

void rotate_matrix(std::vector<std::vector<int>>& matrix) {
    int n = matrix.size(); // Get the number of rows (assuming square matrix)

    // Transpose the matrix
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            std::swap(matrix[i][j], matrix[j][i]);
        }
    }

    // Reverse each row
    for (int i = 0; i < n; ++i) {
        std::reverse(matrix[i].begin(), matrix[i].end());
    }
}

int main() {
    std::vector<std::vector<int>> input_matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    
    rotate_matrix(input_matrix);
    
    std::cout << "Rotated matrix:" << std::endl;
    for (const auto& row : input_matrix) {
        for (int num : row) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }
    
    return 0; // return 0 to indicate the program ended successfully
}""",
            "python_code": """
def rotate_matrix(matrix):
    # Transpose the matrix
    transposed = list(zip(*matrix))
    
    # Reverse each row of the transposed matrix
    rotated = [list(row)[::-1] for row in transposed]
    
    return rotated

# Test the function with the input [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
input_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(rotate_matrix(input_matrix))  # Output: [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
"""
        }
    ]
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(code_pairs, json_file, indent=4)

# Save the code pairs to a JSON file
save_code_pairs_to_json("Hand_code_pairs_Intermediate.json")
