import logging
import argparse
from typing import Dict, Any

from code_parser import CodeParser
from prompt_engineer import PromptEngineer
from model_client import ModelClient
from output_processor import OutputProcessor
from code_translator import CodeTranslator

def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.
    
    Args:
        verbose (bool): Whether to enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def process_file(file_path: str) -> str:
    """
    Read code from a file.
    
    Args:
        file_path (str): Path to the file to read
        
    Returns:
        str: Content of the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")

def save_to_file(output_path: str, content: str) -> None:
    """
    Save content to a file.
    
    Args:
        output_path (str): Path to save the file
        content (str): Content to save
        
    Raises:
        IOError: If there's an error writing to the file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(content)
    except Exception as e:
        raise IOError(f"Error writing to file: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Code Translation Tool")
    
    parser.add_argument("input_file", help="Path to the input code file")
    parser.add_argument("output_file", help="Path to save the translated code")
    parser.add_argument("--source", required=True, help="Source programming language")
    parser.add_argument("--target", required=True, help="Target programming language")
    parser.add_argument("--model", default="llama3.1:latest", help="Model to use for translation")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for model sampling")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

def display_results(results: Dict[str, Any]) -> None:
    """
    Display translation results in a formatted way.
    
    Args:
        results (Dict[str, Any]): Translation results
    """
    if results["success"]:
        print("\n" + "="*50)
        print("TRANSLATION SUCCESSFUL")
        print("="*50)
        
        print("\nCode Structure:")
        for key, value in results["code_structure"].items():
            print(f"  {key}: {value}")
            
        print(f"\nSyntax Valid: {results['syntax_valid']}")
        
        # Preview of translated code (first few lines)
        code_lines = results["formatted_code"].split("\n")
        preview_lines = 10  # Number of lines to preview
        
        print("\nTranslated Code Preview:")
        for i, line in enumerate(code_lines[:preview_lines]):
            print(f"  {i+1}: {line}")
            
        if len(code_lines) > preview_lines:
            print(f"  ... and {len(code_lines) - preview_lines} more lines")
    else:
        print("\n" + "="*50)
        print("TRANSLATION FAILED")
        print("="*50)
        print(f"Error: {results['error']}")

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Read input code
        logger.info(f"Reading code from {args.input_file}")
        code = process_file(args.input_file)
        
        # Initialize model client
        model_client = ModelClient(
            model=args.model,
            temperature=args.temperature
        )
        
        # Initialize translator
        translator = CodeTranslator(
            source_language=args.source,
            target_language=args.target,
            model_client=model_client
        )
        
        # Translate the code
        logger.info(f"Translating from {args.source} to {args.target}")
        results = translator.translate(code)
        
        # Display results
        display_results(results)
        
        # Save translated code if successful
        if results["success"]:
            logger.info(f"Saving translated code to {args.output_file}")
            save_to_file(args.output_file, results["formatted_code"])
            print(f"\nTranslated code saved to {args.output_file}")
            
        return 0
        
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"Error: {str(e)}")
        return 1
    except IOError as e:
        logger.error(str(e))
        print(f"Error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
