import logging
from typing import Dict, Any

from code_parser import CodeParser
from prompt_engineer import PromptEngineer
from model_client import ModelClient
from output_processor import OutputProcessor

logger = logging.getLogger(__name__)

class CodeTranslator:
    """
    Main class that orchestrates the code translation process.
    
    This class brings together parsing, prompt generation, and output processing.
    """
    
    def __init__(
        self, 
        source_language: str, 
        target_language: str,
        model_client: ModelClient = None
    ):
        """
        Initialize the CodeTranslator.
        
        Args:
            source_language (str): Original programming language
            target_language (str): Target programming language
            model_client (ModelClient): Client for model communication
        """
        self.source_language = source_language
        self.target_language = target_language
        self.model_client = model_client or ModelClient()
        
        logger.info(f"Initialized translator for {source_language} to {target_language}")
    
    def translate(self, code: str) -> Dict[str, Any]:
        """
        Translate code from source to target language.
        
        Args:
            code (str): Source code to translate
            
        Returns:
            Dict[str, Any]: Results including translated code and metadata
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If translation fails
        """
        results = {
            "success": False,
            "translated_code": "",
            "formatted_code": "",
            "syntax_valid": False,
            "code_structure": {},
            "error": None
        }
        
        try:
            # 1. Parse the source code
            parser = CodeParser(self.source_language, code)
            code_structure = parser.parse()
            results["code_structure"] = code_structure
            
            # 2. Generate translation prompt
            engineer = PromptEngineer(
                self.source_language,
                self.target_language,
                code_structure,
                code
            )
            
            prompt = engineer.generate_prompt()
            
            # 3. Get translation from model
            translated_code = self.model_client.translate_code(prompt)
            results["translated_code"] = translated_code
            
            # 4. Process and format the output
            processor = OutputProcessor(translated_code, self.target_language)
            formatted_code = processor.format_code()
            results["formatted_code"] = formatted_code
            
            # 5. Verify syntax
            syntax_valid = processor.verify_syntax()
            results["syntax_valid"] = syntax_valid
            
            results["success"] = True
            
        except ValueError as e:
            logger.error(f"Input validation error: {str(e)}")
            results["error"] = f"Validation error: {str(e)}"
        except RuntimeError as e:
            logger.error(f"Runtime error during translation: {str(e)}")
            results["error"] = f"Runtime error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            results["error"] = f"Unexpected error: {str(e)}"
        
        return results
