import os
import subprocess
import json
from pathlib import Path

def run_translation(input_file, output_file, source_lang, target_lang, model="qwen2.5:3b", temperature=0.3, verbose=False):
    """
    Run the code translation tool with the specified parameters.
    
    Args:
        input_file (str): Path to the input code file
        output_file (str): Path to save the translated code
        source_lang (str): Source programming language
        target_lang (str): Target programming language
        model (str): Model to use for translation
        temperature (float): Temperature for model sampling
        verbose (bool): Enable verbose logging
        
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    # Build the command
    cmd = [
        "python", "main.py",  # Assuming the main script is named main.py
        input_file, 
        output_file, 
        "--source", source_lang, 
        "--target", target_lang,
        "--model", model,
        "--temperature", str(temperature)
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    # Run the command
    process = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True
    )
    
    return process.returncode, process.stdout, process.stderr

def verify_output(output_file, expected_output_file=None):
    """
    Verify that the output file exists and optionally compare with expected output.
    
    Args:
        output_file (str): Path to the output file
        expected_output_file (str, optional): Path to the expected output file
        
    Returns:
        dict: Verification results
    """
    result = {
        "exists": os.path.exists(output_file),
        "size": os.path.getsize(output_file) if os.path.exists(output_file) else 0,
        "matches_expected": False
    }
    
    if expected_output_file and os.path.exists(expected_output_file) and result["exists"]:
        with open(output_file, 'r') as f1, open(expected_output_file, 'r') as f2:
            result["matches_expected"] = f1.read() == f2.read()
    
    return result

def run_test_case(test_case):
    """
    Run a single test case and return the results.
    
    Args:
        test_case (dict): Test case configuration
        
    Returns:
        dict: Test results
    """
    print(f"Running test: {test_case['name']}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(test_case['output_file']), exist_ok=True)
    
    # Run the translation
    return_code, stdout, stderr = run_translation(
        test_case['input_file'],
        test_case['output_file'],
        test_case['source_lang'],
        test_case['target_lang'],
        test_case.get('model', "llama3.1:latest"),
        test_case.get('temperature', 0.3),
        test_case.get('verbose', False)
    )
    
    # Verify the output
    verification = verify_output(
        test_case['output_file'], 
        test_case.get('expected_output_file')
    )
    
    # Compile the results
    results = {
        "name": test_case['name'],
        "success": return_code == 0 and verification['exists'],
        "return_code": return_code,
        "verification": verification,
        "stdout": stdout,
        "stderr": stderr
    }
    
    return results

def main():
    """Run all test cases and report results."""
    # Define test cases
    test_cases = [
        {
            "name": "Python to JavaScript Basic",
            "input_file": "test_cases/hello_world.py",
            "output_file": "test_output/hello_world.js",
            "source_lang": "python",
            "target_lang": "javascript",
            "expected_output_file": "test_cases/expected/hello_world.js"
        },
        {
            "name": "Java to C# Complex",
            "input_file": "test_cases/complex_example.java",
            "output_file": "test_output/complex_example.cs",
            "source_lang": "java",
            "target_lang": "csharp",
            "temperature": 0.5
        },
        {
            "name": "Error Handling - Non-existent File",
            "input_file": "test_cases/nonexistent_file.py",
            "output_file": "test_output/should_not_create.js",
            "source_lang": "python",
            "target_lang": "javascript"
        }
    ]
    
    # Run all test cases
    results = []
    for test_case in test_cases:
        result = run_test_case(test_case)
        results.append(result)
        
        # Print result summary
        status = "PASSED" if result["success"] else "FAILED"
        print(f"  {status}: {result['name']}")
    
    # Print overall summary
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"\nTest Summary: {passed}/{total} tests passed")
    
    # Save detailed results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to test_results.json")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())
