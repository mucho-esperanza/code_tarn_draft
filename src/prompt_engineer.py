import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class PromptEngineer:
    """
    Generates prompts for code translation.
    
    Attributes:
        source_language (str): The original programming language
        target_language (str): The language to translate to
        code_structure (dict): Parsed structure of the code
        code (str): The original source code
    """
    
    def __init__(
        self, 
        source_language: str, 
        target_language: str, 
        code_structure: Dict[str, List[str]], 
        code: str
    ):
        """
        Initialize the PromptEngineer with translation parameters.
        
        Args:
            source_language (str): Original programming language
            target_language (str): Target programming language
            code_structure (dict): Parsed structure of the code
            code (str): Original source code
        """
        self.source_language = source_language
        self.target_language = target_language
        self.code_structure = code_structure
        self.code = code
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.code.strip():
            raise ValueError("Empty code provided for translation")
    
    def generate_prompt(self) -> str:
        """
        Generate a prompt for the code translation model.
        
        Returns:
            str: The formatted prompt
        """
        # Extract key information from code structure for the prompt
        functions_str = ", ".join(self.code_structure.get("functions", []))
        classes_str = ", ".join(self.code_structure.get("classes", []))
        dependencies_str = ", ".join(self.code_structure.get("dependencies", []))
        challenges_str = ", ".join(self.code_structure.get("challenges", []))
        
        prompt = f"""
You are a precise code translation expert. Your task is to translate code between programming languages while maintaining functionality, readability, and adherence to best practices. Follow these guidelines:

1. **Accuracy:** Ensure the translated code performs the same functionality as the original.
2. **Readability:** Use proper indentation, comments, and variable naming conventions.
3. **Best Practices:** Follow the idiomatic style and conventions of the target language.
4. **Error Handling:** Include appropriate error handling if applicable.
5. **Output Format:** Provide **only the translated code** without any additional explanations, comments, or pre/post text. The output must be ready to use in an IDE.

Source Language: {self.source_language}
Target Language: {self.target_language}

Code Structure Information:
- Functions: {functions_str}
- Classes: {classes_str}
- Dependencies: {dependencies_str}
- Potential Challenges: {challenges_str}

Original Code:
```{self.source_language}
{self.code}
```

Translate the above code to {self.target_language} following the target language's best practices.
"""
        return prompt.strip()
