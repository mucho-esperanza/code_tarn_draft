import json

def save_code_pairs_to_json(file_path):
    code_pairs = [
        {
            "title": "01 Multithreading with Shared Resources",
            "cpp_code": """
#include <iostream>
#include <thread>
#include <mutex>

std::mutex mtx;

void incrementCounter(int& counter, int iterations) {
    for (int i = 0; i < iterations; i++) {
        std::lock_guard<std::mutex> lock(mtx);
        counter++;
    }
}

int main() {
    int counter = 0;
    int iterations = 1000;
    
    std::thread t1(incrementCounter, std::ref(counter), iterations);
    std::thread t2(incrementCounter, std::ref(counter), iterations);
    
    t1.join();
    t2.join();

    std::cout << "Final counter value: " << counter << std::endl;
    return 0;
}""",
            "python_code": """
import threading

counter_lock = threading.Lock()

def increment_counter(counter, iterations):
    for _ in range(iterations):
        with counter_lock:
            counter[0] += 1

def main():
    counter = [0]
    iterations = 1000

    t1 = threading.Thread(target=increment_counter, args=(counter, iterations))
    t2 = threading.Thread(target=increment_counter, args=(counter, iterations))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Final counter value:", counter[0])

main()
"""
        },
        {
            "title": "02 Template Function with Edge Cases",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <typeinfo>

template <typename T>
void printElements(std::vector<T> vec) {
    for (auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> intVec = {1, 2, 3, 4, 5};
    std::vector<std::string> strVec = {"hello", "world"};

    printElements(intVec);
    printElements(strVec);
    return 0;
}""",
            "python_code": """
def print_elements(vec):
    for elem in vec:
        print(elem, end=" ")
    print()

def main():
    int_vec = [1, 2, 3, 4, 5]
    str_vec = ["hello", "world"]

    print_elements(int_vec)
    print_elements(str_vec)

main()
"""
        },
        {
            "title": "03 Dynamic Memory Management with Shared Pointers",
            "cpp_code": """
#include <iostream>
#include <memory>

class Node {
public:
    int value;
    std::shared_ptr<Node> next;

    Node(int val) : value(val), next(nullptr) {}
};

void printList(std::shared_ptr<Node> head) {
    while (head) {
        std::cout << head->value << " -> ";
        head = head->next;
    }
    std::cout << "null" << std::endl;
}

int main() {
    std::shared_ptr<Node> head = std::make_shared<Node>(1);
    head->next = std::make_shared<Node>(2);
    head->next->next = std::make_shared<Node>(3);

    printList(head);

    return 0;
}""",
            "python_code": """
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

def print_list(head):
    while head:
        print(f"{head.value} -> ", end="")
        head = head.next
    print("null")

def main():
    head = Node(1)
    head.next = Node(2)
    head.next.next = Node(3)

    print_list(head)

main()
"""
        },
        {
            "title": "04 Lambda Functions with Captures",
            "cpp_code": """
#include <iostream>
#include <vector>
#include <algorithm>

void filterAndPrint(std::vector<int> vec, int threshold) {
    auto lambda = [threshold](int val) {
        return val > threshold;
    };
    vec.erase(std::remove_if(vec.begin(), vec.end(), lambda), vec.end());
    for (int val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> vec = {1, 2, 3, 4, 5, 6};
    filterAndPrint(vec, 3);
    return 0;
}""",
            "python_code": """
def filter_and_print(vec, threshold):
    lambda_func = lambda val: val > threshold
    vec = list(filter(lambda_func, vec))
    for val in vec:
        print(val, end=" ")
    print()

def main():
    vec = [1, 2, 3, 4, 5, 6]
    filter_and_print(vec, 3)

main()
"""
        }
    ]
    
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(code_pairs, json_file, indent=4)

# Save the code pairs to a JSON file
save_code_pairs_to_json("code_pairs_Complex.json")
