import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ModelClient:
    """
    Client for interacting with language models for code translation.
    
    This class handles communication with the language model service.
    """
    
    def __init__(
        self, 
        model: str = "codellama", 
        temperature: float = 0.3, 
        **kwargs
    ):
        """
        Initialize the ModelClient.
        
        Args:
            model (str): LLM model to use
            temperature (float): Temperature setting for the model
            **kwargs: Additional parameters for the model
        """
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
        
        # Validate settings
        if not 0 <= self.temperature <= 1:
            logger.warning(f"Temperature {self.temperature} outside recommended range [0-1]")
    
    def translate_code(self, prompt: str) -> str:
        """
        Send the prompt to the model and return translated code.
        
        Args:
            prompt (str): The prompt to send to the model
            
        Returns:
            str: The translated code
            
        Raises:
            RuntimeError: If there's an error communicating with the model
        """
        try:
            import ollama
            
            # Set timeout to prevent hanging
            timeout = self.kwargs.pop("timeout", 30)
            
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                options={"temperature": self.temperature, **self.kwargs}
            )
            
            translated_code = response.get("message", {}).get("content", "")
            
            # Basic validation of response
            if not translated_code:
                logger.warning("Received empty translation from model")
                return ""
                
            # Extract code from markdown code blocks if present
            code_block_pattern = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)
            code_blocks = code_block_pattern.findall(translated_code)
            
            if code_blocks:
                # Use the longest code block found
                translated_code = max(code_blocks, key=len)
            
            return translated_code
            
        except ImportError:
            logger.error("Ollama module not installed. Install with: pip install ollama")
            raise RuntimeError("Ollama module is required but not installed")
        except Exception as e:
            logger.error(f"Error calling language model: {str(e)}")
            raise RuntimeError(f"Failed to get translation: {str(e)}")
