from code_parser import CodeParser
from prompt_engineer import PromptEngineer
from model_client import ModelClient
from output_processor import OutputProcessor
from code_translator import CodeTranslator

def simple_example():
    """Example usage of the code translator with direct API calls."""
    # Define the code sample
    code_sample = """
    #include <iostream>
    using namespace std;
    
    class Test { 
    public: 
        void func() {
            cout << "Test function" << endl;
        } 
    };
    
    int main() { 
        Test t;
        t.func();
        if (true) { 
            cout << "Hello World!" << endl; 
        } 
        return 0; 
    }
    """
    
    # Create a translator instance
    model_client = ModelClient(model="qwen2.5:3b", temperature=0.3)
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
