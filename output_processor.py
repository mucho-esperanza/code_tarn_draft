import re
import ast
import logging

logger = logging.getLogger(__name__)

class OutputProcessor:
    """
    Processes and formats translated code.
    
    Attributes:
        translated_code (str): The translated code to process
        target_language (str): The target programming language
    """
    
    def __init__(self, translated_code: str, target_language: str):
        """
        Initialize the OutputProcessor.
        
        Args:
            translated_code (str): Translated code to process
            target_language (str): Target programming language
        """
        self.translated_code = translated_code
        self.target_language = target_language.lower()
        
        # Common formatters for different languages
        self.formatters = {
            "python": self._format_python,
            "javascript": self._format_javascript,
            "java": self._format_java
        }
    
    def format_code(self) -> str:
        """
        Format code according to target language conventions.
        
        Returns:
            str: Formatted code
        """
        # Apply specific formatter if available
        if self.target_language in self.formatters:
            self.translated_code = self.formatters[self.target_language]()
        
        return self.translated_code
    
    def _format_python(self) -> str:
        """Format Python code."""
        # Convert tabs to spaces
        code = re.sub(r'\t', '    ', self.translated_code)
        
        # Remove trailing whitespace
        code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)
        
        # Ensure two blank lines before class/function definitions
        code = re.sub(r'(\n+)class', r'\n\n\nclass', code)
        code = re.sub(r'(\n+)def', r'\n\n\ndef', code)
        
        # Fix multiple blank lines
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        return code
    
    def _format_javascript(self) -> str:
        """Format JavaScript code."""
        # Basic JS formatting
        return self.translated_code
    
    def _format_java(self) -> str:
        """Format Java code."""
        # Basic Java formatting
        return self.translated_code
    
    def add_comments(self, name: str = "", author: str = "", date: str = "") -> str:
        """
        Add comments to the translated code.
        
        Args:
            name (str): Name of the code file
            author (str): Author of the translation
            date (str): Date of translation
            
        Returns:
            str: Code with added comments
        """
        comment_start = ""
        
        # Set language-specific comment markers
        if self.target_language == "python":
            comment_start = "# "
        elif self.target_language in ["javascript", "java", "c++", "c", "cpp"]:
            comment_start = "// "
        
        # Add header comment
        header = [
            f"{comment_start}Translated code",
            f"{comment_start}Original language: {self.target_language.capitalize()}"
        ]
        
        if name:
            header.append(f"{comment_start}Name: {name}")
        if author:
            header.append(f"{comment_start}Translated by: {author}")
        if date:
            header.append(f"{comment_start}Date: {date}")
            
        header.append(f"{comment_start}{'-' * 50}")
        header.append("")
        
        return "\n".join(header) + self.translated_code
    
    def verify_syntax(self) -> bool:
        """
        Check syntax validity without executing code.
        
        Returns:
            bool: True if syntax is valid, False otherwise
        """
        if not self.translated_code.strip():
            logger.warning("Empty code provided for syntax verification")
            return False
            
        if self.target_language == "python":
            try:
                ast.parse(self.translated_code)
                return True
            except SyntaxError as e:
                line_num = e.lineno if hasattr(e, 'lineno') else 'unknown'
                logger.error(f"Python syntax error at line {line_num}: {str(e)}")
                return False
                
        elif self.target_language in ["javascript", "java", "c++", "c", "cpp"]:
            # For non-Python languages, we can only do basic checks
            # A proper implementation would use language-specific tools
            logger.info(f"Full syntax validation for {self.target_language} not implemented")
            
            # Basic check for balanced braces, brackets, and parentheses
            return self._check_balanced_symbols()
            
        return True
    
    def _check_balanced_symbols(self) -> bool:
        """
        Check if code has balanced braces, brackets, and parentheses.
        
        Returns:
            bool: True if balanced, False otherwise
        """
        stack = []
        opening = "({["
        closing = ")}]"
        
        # Strip string literals and comments to avoid false positives
        # This is a simplification and not a full parser
        code = re.sub(r'"([^"\\]|\\.)*"', '""', self.translated_code)
        code = re.sub(r"'([^'\\]|\\.)*'", "''", code)
        
        for char in code:
            if char in opening:
                stack.append(char)
            elif char in closing:
                if not stack:
                    return False
                if opening.index(stack.pop()) != closing.index(char):
                    return False
                    
        return len(stack) == 0
