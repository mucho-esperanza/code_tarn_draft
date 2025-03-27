# CodeParser: A Versatile Code Structural Analysis Tool

## Key Features
- Supports parsing of Python and C++ code
- Extracts structural information from source code
- Identifies key code elements like:
  - Functions
  - Classes
  - Control flow statements
  - Dependencies
  - Potential code challenges

## Core Functionality
The `CodeParser` class is designed to analyze source code and break it down into meaningful components:
1. Supports multiple programming languages (currently Python and C++)
2. Uses different parsing strategies:
   - Python: Utilizes Abstract Syntax Tree (AST) for deep code analysis
   - C++: Employs regular expressions for structural extraction

## Parsing Capabilities
- For Python:
  - Identifies function and class definitions
  - Detects control flow structures
  - Extracts import dependencies
  - Recognizes advanced language features like:
    - Asynchronous functions
    - List/dict comprehensions
    - Exception handling

- For C++:
  - Extracts function and class names
  - Identifies include dependencies
  - Detects control flow statements
  - Flags potential challenges like:
    - Pointer management
    - Manual memory management
    - Template usage
    - Multithreading complexities

## Error Handling
- Comprehensive logging
- Graceful error handling
- Supports empty code inputs
- Provides informative error messages

## Testing
Includes built-in test cases for both Python and C++ parsing, demonstrating the functionality and validating the parser's capabilities.
