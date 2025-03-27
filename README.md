# Code Translation Tool 🖥️🔄

## Overview
The **Code Translation Tool** is a powerful command-line utility designed to translate source code between different programming languages using a large language model (LLM). By leveraging advanced prompt engineering techniques, the tool ensures high-quality, accurate code translations.

## 🌟 Key Features
- **Cross-Language Translation**: Seamlessly convert code between various programming languages
- **Local Model Support**: Utilizes **Ollama** as a flexible, local model provider
- **Customizable Models**: Select from a wide range of Ollama's available language models
- **Code Quality Assurance**: Includes syntax validation and structured code formatting
- **Comprehensive Logging**: Provides detailed logging and verbose output for easy debugging

## 🛠 Requirements
- Python 3.12+
- Ollama (installed and running locally)
- Project dependencies (detailed in installation steps)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mucho-esperanza/code_tarn_draft.git
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Ollama
- Download and install Ollama from the [official website](https://ollama.com)
- Pull a code translation model:
```bash
ollama pull qwen2.5-coder:3b  # Example model
```
- Start Ollama service:
```bash
ollama serve
```

## 🎬 Quick Start

### Command Syntax
```bash
python main.py <input_file> <output_file> <source_lang> <target_lang> [options]
```

### Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `input_file` | Path to input code file (JSON) | Required |
| `output_file` | Path to save translated code (JSON) | Required |
| `source_lang` | Source programming language | Required |
| `target_lang` | Target programming language | Required |
| `--model` | Model for translation | `codellama:latest` |
| `--temperature` | Model response sampling temperature | `0.3` |
| `--verbose` | Enable detailed logging | `False` |

### Example Usage
```bash
# Translate C++ code to Python with verbose logging
python main.py examples/cpp_code.json output/translated_python.json "C++" "Python" --model "qwen2.5-coder" --verbose
```

## 🤖 Model Recommendations
- **Ensure Ollama is installed and running before use**
- Recommended models for code translation:
```bash
ollama pull codellama:latest
ollama pull qwen2.5-coder:3b
```

## 📝 Logging & Debugging
- Use `--verbose` flag to enable detailed logging
- Log format: `YYYY-MM-DD HH:MM:SS,mmm - Module - LEVEL - Message`

## 📄 License
MIT License - Open source and free to use

## 🤝 Support
- [Open an issue](https://github.com/yourusername/code-translation-tool/issues) on GitHub
- Contact the maintainer for further assistance

## 💡 Contributing
Contributions are welcome! Please read the contribution guidelines before getting started.
