import logging
import json
#from model_client import ModelClient
#from code_translator import CodeTranslator
from src.code_translator import CodeTranslator
from src.model_client import ModelClient
from src.code_parser import CodeParser
from src.code_translator import CodeTranslator

def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_test_cases(file_path: str):
    """Load test cases from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_results(results, output_path: str):
    """Save translation results to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4)

def run_translation_pipeline(test_cases):
    """
    Runs the translation pipeline for each test case and saves results.
    """
    model_client = ModelClient(model="deepseek-r1:1.5b", temperature=0.3)
    translator = CodeTranslator(source_language="cpp", target_language="python", model_client=model_client)
    
    results = []
    for test in test_cases:
        title = test["title"]
        cpp_code = test["cpp_code"].strip()
        expected_python_code = test["python_code"].strip()
        
        try:
            translation_result = translator.translate(cpp_code)
            success = translation_result["success"]
            translated_code = translation_result["formatted_code"].strip()
            
            comparison = translated_code == expected_python_code
            
            print("\n" + "="*50)
            print(f"Title: {title}")
            print("="*50)
            print("\nSource C++ Code:")
            print(cpp_code)
            print("\nReference Python Code:")
            print(expected_python_code)
            print("\nGenerated Python Code:")
            print(translated_code)
            print("="*50 + "\n")
            
            results.append({
                "title": title,
                "cpp_code": cpp_code,
                "translated_python_code": translated_code,
                "expected_python_code": expected_python_code,
                "match_with_reference": comparison,
                "success": success,
                "error": translation_result.get("error", None)
            })
        
        except Exception as e:
            results.append({
                "title": title,
                "cpp_code": cpp_code,
                "translated_python_code": None,
                "expected_python_code": expected_python_code,
                "match_with_reference": False,
                "success": False,
                "error": str(e)
            })
    
    return results

def main():
    setup_logging(verbose=True)
    logger = logging.getLogger(__name__)
    
    test_cases_file = "Hand_code_pairs_Advance.json"
    results_file = "Result_Hand_code_pairs_Advance_deepseek-r1.json"
    
    logger.info(f"Loading test cases from {test_cases_file}")
    test_cases = load_test_cases(test_cases_file)
    
    logger.info("Running translation pipeline...")
    results = run_translation_pipeline(test_cases)
    
    logger.info(f"Saving results to {results_file}")
    save_results(results, results_file)
    
    print(f"Translation results saved to {results_file}")
    
if __name__ == "__main__":
    main()
