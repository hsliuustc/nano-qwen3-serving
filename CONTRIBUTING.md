# Contributing to Nano Qwen3 Serving

Thank you for your interest in contributing to Nano Qwen3 Serving! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **🐛 Bug Reports**: Report issues you encounter
- **✨ Feature Requests**: Suggest new features or improvements
- **📝 Documentation**: Improve or add documentation
- **🔧 Code Contributions**: Submit code changes
- **🧪 Testing**: Add tests or improve test coverage
- **🌐 Translations**: Help with internationalization

### Before You Start

1. **Check Existing Issues**: Search existing issues to avoid duplicates
2. **Read Documentation**: Familiarize yourself with the project structure
3. **Set Up Development Environment**: Follow the setup instructions below

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Apple Silicon Mac (M1/M2/M3) with macOS 12.3+
- Git

### Local Development

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/nano-qwen3-serving.git
cd nano-qwen3-serving

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests to ensure everything works
python -m pytest tests/
```

### Code Style

We follow PEP 8 style guidelines. Please ensure your code:

- Uses 4 spaces for indentation
- Has proper docstrings
- Follows naming conventions
- Is properly formatted

You can use tools to help:

```bash
# Install development tools
pip install black flake8 isort

# Format code
black nano_qwen3_serving/

# Sort imports
isort nano_qwen3_serving/

# Check style
flake8 nano_qwen3_serving/
```

## 📝 Making Changes

### 1. Create a Branch

```bash
# Create a new branch for your feature
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-description
```

### 2. Make Your Changes

- Write clear, well-documented code
- Add tests for new functionality
- Update documentation if needed
- Follow the existing code patterns

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/

# Run specific tests
python -m pytest tests/test_specific_feature.py

# Run with coverage
python -m pytest tests/ --cov=nano_qwen3_serving

# Test the service
python tools/start_service.py --port 8001 &
curl -X GET http://127.0.0.1:8001/health
```

### 4. Commit Your Changes

```bash
# Add your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### 5. Create a Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template
5. Submit the PR

## 📋 Pull Request Guidelines

### PR Template

When creating a PR, please include:

- **Description**: What does this PR do?
- **Type**: Bug fix, feature, documentation, etc.
- **Testing**: How was this tested?
- **Breaking Changes**: Any breaking changes?
- **Related Issues**: Link to related issues

### PR Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation is updated
- [ ] No breaking changes (or documented)
- [ ] Commit messages are clear
- [ ] PR description is complete

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_llm.py

# Run with coverage
python -m pytest --cov=nano_qwen3_serving --cov-report=html
```

### Writing Tests

- Write tests for new functionality
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies when appropriate

Example test structure:

```python
def test_feature_name():
    """Test description of what is being tested."""
    # Arrange
    expected = "expected result"
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected
```

## 📚 Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google or NumPy docstring style
- Include examples in docstrings

### API Documentation

- Update API documentation for new endpoints
- Include request/response examples
- Document all parameters and return values

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, dependencies
2. **Steps to Reproduce**: Clear, step-by-step instructions
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Error Messages**: Full error traceback
6. **Additional Context**: Any relevant information

## ✨ Feature Requests

When requesting features, please include:

1. **Problem**: What problem does this solve?
2. **Solution**: How should it work?
3. **Use Cases**: Who would benefit?
4. **Alternatives**: Any existing solutions?

## 🏷️ Issue Labels

We use the following labels:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## 📞 Getting Help

If you need help:

1. **Check Documentation**: Read the README and docs
2. **Search Issues**: Look for similar questions
3. **Ask Questions**: Open a new issue with the `question` label
4. **Join Discussions**: Use GitHub Discussions

## 🎉 Recognition

Contributors will be recognized in:

- The project README
- Release notes
- GitHub contributors page

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Nano Qwen3 Serving! 🚀 