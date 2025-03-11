import json

def save_code_pairs_to_json(file_path):
    code_pairs = [
        {
            "title": "Basic array operations",
            "cpp_code": """
vector<int> reverseArray(vector<int>& arr) {
    vector<int> result;
    for(int i = arr.size() - 1; i >= 0; i--) {
        result.push_back(arr[i]);
    }
    return result;
}""",
            "python_code": """
def reverse_array(arr):
    return arr[::-1]
"""
        },
        {
            "title": "String manipulation",
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
        },
        {
            "title": "Mathematical operations",
            "cpp_code": """
int factorial(int n) {
    if(n <= 1) return 1;
    return n * factorial(n-1);
}""",
            "python_code": """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
        },
        {
            "title": "Search algorithm",
            "cpp_code": """
int binarySearch(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if(arr[mid] == target) return mid;
        if(arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}""",
            "python_code": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
        },
        {
            "title": "Data structure operations",
            "cpp_code": """
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(NULL) {}
};

ListNode* reverseLinkedList(ListNode* head) {
    ListNode *prev = NULL;
    ListNode *curr = head;
    while(curr != NULL) {
        ListNode *next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}""",
            "python_code": """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_linked_list(head):
    prev = None
    curr = head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev
"""
        }
    ]
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(code_pairs, json_file, indent=4)

# Save the code pairs to a JSON file
save_code_pairs_to_json("code_pairs_simplecodes.json")
