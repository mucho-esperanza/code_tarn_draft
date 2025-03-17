#from src import CodeParser, PromptEngineer, ModelClient, OutputProcessor, CodeTranslator
from src.code_parser import CodeParser
from src.prompt_engineer import PromptEngineer
from src.model_client import ModelClient
from src.output_processor import OutputProcessor
from src.code_translator import CodeTranslator

def simple_example():
    """Example usage of the code translator with direct API calls."""
    # Define the code sample
    code_sample = """
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
}
    """
    
    # Create a translator instance
    model_client = ModelClient(model="codellama:latest", temperature=0.3)
    translator = CodeTranslator("C++", "Python", model_client)
    
    # Translate the code
    results = translator.translate(code_sample)
    
    # Display results
    if results["success"]:
        print("Translation successful!")
        print("\nCode Structure:")
        for key, value in results["code_structure"].items():
            print(f"  {key}: {value}")
            
        print("\nTranslated Code:")
        print(results["formatted_code"])
        
        print(f"\nSyntax Valid: {results['syntax_valid']}")
    else:
        print(f"Translation failed: {results['error']}")

if __name__ == "__main__":
    simple_example()