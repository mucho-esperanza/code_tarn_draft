import re
import ast
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeParser:
    """
    Parses code in various programming languages and extracts structural information.
    
    Attributes:
        language (str): The programming language of the code to parse
        code (str): The source code to be parsed
        parsed_data (Dict): Dictionary containing the parsed code structure
    """
    
    def __init__(self, language: str, code: str):
        """
        Initialize the CodeParser with the specified language and code.
        
        Args:
            language (str): Programming language of the code
            code (str): Source code to be parsed
        """
        self.language = language.lower()
        self.code = code
        self.parsed_data = {
            "functions": [],
            "classes": [],
            "control_flow": [],
            "dependencies": [],
            "challenges": []
        }
        
        # Pre-compile regex patterns for efficiency
        self._cpp_patterns = {
            "functions": re.compile(r'(?<!\S)(?!if|for|while|switch)[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(?={)'),
            "classes": re.compile(r'\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b'),
            "dependencies": re.compile(r'#include\s+[<"]([a-zA-Z0-9_\.]+)[>"]'),
            "control_flow": re.compile(r'\b(if|for|while|switch|try|catch)\s*\(')
        }
    
    def parse(self) -> Dict[str, List[str]]:
        """
        Parse the code according to its language.
        
        Returns:
            Dict[str, List[str]]: Dictionary containing the parsed code structure
            
        Raises:
            ValueError: If the language is not supported or the code is invalid
        """
        if not self.code.strip():
            logger.warning("Empty code string provided for parsing")
            return self.parsed_data
            
        try:
            if self.language == "python":
                self._parse_python()
            elif self.language in ["cpp", "c++", "c"]:
                self._parse_cpp()
            else:
                supported = ["python", "cpp", "c++", "c"]
                raise ValueError(f"Unsupported language. Supported languages: {', '.join(supported)}")
        except Exception as e:
            logger.error(f"Error parsing {self.language} code: {str(e)}")
            raise ValueError(f"Failed to parse {self.language} code: {str(e)}")
            
        return self.parsed_data
    
    def _parse_python(self) -> None:
        """Parse Python code using the abstract syntax tree."""
        try:
            tree = ast.parse(self.code)
            
            # Single walk through the AST for efficiency
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.parsed_data["functions"].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    self.parsed_data["classes"].append(node.name)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    self.parsed_data["control_flow"].append(type(node).__name__.lower())
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if hasattr(node, 'module') and node.module:
                            # Handle from X import Y
                            module_prefix = f"{node.module}." if node.level == 0 else "."*node.level
                            self.parsed_data["dependencies"].append(f"{module_prefix}{alias.name}")
                        else:
                            # Handle direct imports
                            self.parsed_data["dependencies"].append(alias.name)
                            
            # Identify potential challenges
            if any(isinstance(node, ast.AsyncFunctionDef) for node in ast.walk(tree)):
                self.parsed_data["challenges"].append("Asynchronous code")
                
            if any(isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)) 
                  for node in ast.walk(tree)):
                self.parsed_data["challenges"].append("Comprehensions")
                
            if any(isinstance(node, ast.Try) for node in ast.walk(tree)):
                self.parsed_data["challenges"].append("Exception handling")
                
        except SyntaxError as e:
            logger.error(f"Python syntax error: {str(e)}")
            raise ValueError(f"Invalid Python code: {str(e)}")
    
    def _parse_cpp(self) -> None:
        """Parse C++ code using regular expressions."""
        # Extract basic structures using regex
        self.parsed_data["functions"] = self._cpp_patterns["functions"].findall(self.code)
        self.parsed_data["classes"] = self._cpp_patterns["classes"].findall(self.code)
        self.parsed_data["dependencies"] = self._cpp_patterns["dependencies"].findall(self.code)
        
        # Extract control flow statements
        control_flow_matches = self._cpp_patterns["control_flow"].findall(self.code)
        self.parsed_data["control_flow"] = list(set(control_flow_matches))  # Remove duplicates

        # Identify potential challenges
        if "*" in self.code or "&" in self.code:
            self.parsed_data["challenges"].append("Pointer management")
            
        if "new" in self.code or "delete" in self.code or "malloc" in self.code or "free" in self.code:
            self.parsed_data["challenges"].append("Manual memory management")
            
        if "template" in self.code or "<typename" in self.code:
            self.parsed_data["challenges"].append("Templates")
            
        if "thread" in self.code or "mutex" in self.code or "atomic" in self.code:
            self.parsed_data["challenges"].append("Multithreading")
