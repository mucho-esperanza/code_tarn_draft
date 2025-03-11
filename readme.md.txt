# Code Translation System

A modular system for translating code between programming languages using language models.

## Overview

This system allows you to translate code from one programming language to another while maintaining functionality and idiomatic style. It uses the following components:

1. **CodeParser**: Analyzes source code to extract structure information
2. **PromptEngineer**: Generates optimized prompts for language models
3. **ModelClient**: Handles communication with the language model
4. **OutputProcessor**: Formats and validates the translated code
5. **CodeTranslator**: Orchestrates the entire translation process

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/code-translator.git
   cd code-translator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have Ollama installed and running: [Ollama Installation Guide](https://github.com/ollama/ollama)

## Usage

### Command Line Interface

The tool can be used from the command line:

```bash
python main.py input.cpp output.py --source C++ --target Python --model llama3.1:latest
```

Arguments:
- `input_file`: Path to the source code file
- `output_file`: Path to save the translated code
- `--source`: Source programming language
- `--target`: Target programming language
- `--model`: Language model to use (default: llama3.1:latest)
- `--temperature`: Model temperature (default: 0.3)
- `--verbose`: Enable verbose logging

### Programmatic Usage

You can also use the system programmatically:

```python
from code_translator import CodeTranslator
from model_client import ModelClient

# Initialize the model client
model_client = ModelClient(model="llama3.1:latest", temperature=0.3)

# Initialize the translator
translator = CodeTranslator("C++", "Python", model_client)

# Translate code
results = translator.translate(code_string)

# Access results
if results["success"]:
    translated_code = results["formatted_code"]
    print(translated_code)
else:
    print(f"Error: {results['error']}")
```

See `example_usage.py` for more detailed examples.

## Supported Languages

- Source Languages: Python, C++
- Target Languages