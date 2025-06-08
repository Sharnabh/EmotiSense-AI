# 🤝 Contributing to EmotiSense-AI

Thank you for your interest in contributing to **EmotiSense-AI**! We welcome contributions from developers, researchers, and ML enthusiasts of all skill levels. This guide will help you get started with contributing to our emotion classification project.

## 📋 Table of Contents

- [🎯 Ways to Contribute](#-ways-to-contribute)
- [🚀 Getting Started](#-getting-started)
- [🔧 Development Setup](#-development-setup)
- [📝 Contributing Guidelines](#-contributing-guidelines)
- [🐛 Reporting Issues](#-reporting-issues)
- [💡 Feature Requests](#-feature-requests)
- [🔄 Pull Request Process](#-pull-request-process)
- [📊 Code Standards](#-code-standards)
- [🧪 Testing Guidelines](#-testing-guidelines)
- [📚 Documentation](#-documentation)
- [🏆 Recognition](#-recognition)

---

## 🎯 Ways to Contribute

### 🐛 **Bug Fixes**
- Fix model loading issues
- Resolve UI/UX problems in Streamlit app
- Address performance bottlenecks
- Correct documentation errors

### ✨ **New Features**
- Additional emotion classification models
- New visualization types
- Enhanced web interface components
- Multi-language support
- API endpoints
- Mobile app development

### 📊 **Model Improvements**
- Hyperparameter optimization
- New architectures (BERT, RoBERTa, etc.)
- Ensemble methods
- Domain-specific fine-tuning
- Model compression techniques

### 📖 **Documentation**
- Improve README sections
- Add code comments
- Create tutorials and examples
- Write blog posts about findings
- Translate documentation

### 🧪 **Testing & Quality**
- Write unit tests
- Add integration tests
- Performance benchmarking
- Code review and refactoring

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.10 or higher
- **Git**: For version control
- **GitHub Account**: To submit pull requests
- **Basic ML Knowledge**: Understanding of NLP and transformers

### 1. Fork the Repository
```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/EmotiSense-AI.git
cd EmotiSense-AI

# Add upstream remote
git remote add upstream https://github.com/Sharnabh/EmotiSense-AI.git
```

### 2. Create a Feature Branch
```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

---

## 🔧 Development Setup

### 1. Environment Setup
```bash
# Create virtual environment
conda create -n emotisense-dev python=3.10
conda activate emotisense-dev

# Or using venv
python -m venv emotisense-dev
# Windows
emotisense-dev\Scripts\activate
# macOS/Linux
source emotisense-dev/bin/activate
```

### 2. Install Dependencies
```bash
# Install main dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Download Required Models
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 4. Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

### 5. Launch Development Environment
```bash
# Start Jupyter for notebook development
jupyter notebook

# Launch Streamlit app
streamlit run app/streamlit_app.py

# Run linting
flake8 src/ tests/
black src/ tests/ --check
```

---

## 📝 Contributing Guidelines

### 🎯 **Code Quality Standards**

#### **Python Style Guide**
- Follow **PEP 8** style guidelines
- Use **Black** for code formatting
- Maximum line length: **88 characters**
- Use **type hints** where appropriate
- Write **docstrings** for all functions/classes

#### **Example Code Style**
```python
from typing import List, Dict, Optional
import torch
import numpy as np


def predict_emotions(
    texts: List[str], 
    model: torch.nn.Module, 
    tokenizer,
    batch_size: int = 16
) -> List[Dict[str, float]]:
    """
    Predict emotions for a batch of text inputs.
    
    Args:
        texts: List of input texts to classify
        model: Trained emotion classification model
        tokenizer: Tokenizer for text preprocessing
        batch_size: Number of texts to process simultaneously
        
    Returns:
        List of emotion predictions with confidence scores
        
    Example:
        >>> predictions = predict_emotions(
        ...     ["I'm so happy!", "This is terrible"], 
        ...     model, 
        ...     tokenizer
        ... )
        >>> print(predictions[0])
        {'joy': 0.95, 'sadness': 0.02, 'anger': 0.01, ...}
    """
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # ... implementation
        
    return predictions
```

### 🧪 **Testing Standards**

#### **Test Coverage Requirements**
- **Minimum coverage**: 80% for new code
- **Critical functions**: 100% coverage required
- **Integration tests**: For end-to-end workflows
- **Performance tests**: For model inference speed

#### **Test Structure**
```python
import pytest
import torch
from src.models.emotion_classifier import EmotionClassifier


class TestEmotionClassifier:
    @pytest.fixture
    def classifier(self):
        """Setup classifier for testing."""
        return EmotionClassifier.load_pretrained('outputs/emotion_classification_model')
    
    def test_single_prediction(self, classifier):
        """Test single text prediction."""
        result = classifier.predict("I'm so excited!")
        
        assert isinstance(result, dict)
        assert 'emotion' in result
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
        
    def test_batch_prediction(self, classifier):
        """Test batch prediction performance."""
        texts = ["Happy text", "Sad text", "Angry text"] * 10
        results = classifier.predict_batch(texts)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, dict) for r in results)
        
    @pytest.mark.slow
    def test_model_loading_speed(self):
        """Test model loading performance."""
        import time
        start_time = time.time()
        
        classifier = EmotionClassifier.load_pretrained('outputs/emotion_classification_model')
        load_time = time.time() - start_time
        
        assert load_time < 10.0  # Should load within 10 seconds
```

### 📊 **Documentation Standards**

#### **Docstring Format**
Use **Google-style docstrings**:

```python
def fine_tune_model(
    dataset_path: str,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    learning_rate: float = 2e-5
) -> Dict[str, float]:
    """
    Fine-tune a transformer model for emotion classification.
    
    This function loads a pre-trained transformer model and fine-tunes it
    on the emotion classification dataset using the Hugging Face Transformers
    library.
    
    Args:
        dataset_path: Path to the emotion classification dataset
        model_name: Name of the pre-trained model to use
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        
    Returns:
        Dictionary containing training metrics:
            - 'accuracy': Final validation accuracy
            - 'f1_score': Macro F1 score
            - 'loss': Final validation loss
            
    Raises:
        FileNotFoundError: If dataset_path doesn't exist
        ValueError: If num_epochs <= 0 or learning_rate <= 0
        
    Example:
        >>> metrics = fine_tune_model(
        ...     "data/emotion_dataset.csv",
        ...     model_name="distilbert-base-uncased",
        ...     num_epochs=3,
        ...     learning_rate=2e-5
        ... )
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        Accuracy: 0.935
        
    Note:
        This function requires a GPU for reasonable training times.
        Training on CPU may take several hours.
    """
    # Implementation here
    pass
```

---

## 🐛 Reporting Issues

### 📝 **Issue Template**

When reporting bugs, please use this template:

```markdown
## 🐛 Bug Report

### **Description**
A clear and concise description of the bug.

### **Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

### **Expected Behavior**
What you expected to happen.

### **Actual Behavior**
What actually happened.

### **Screenshots**
If applicable, add screenshots to help explain your problem.

### **Environment**
- OS: [e.g., Windows 11, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g., 3.10.2]
- Package Versions:
  - torch: [version]
  - transformers: [version]
  - streamlit: [version]

### **Additional Context**
Add any other context about the problem here.

### **Logs**
```
Paste relevant error logs here
```

#### **Priority Levels**
- 🔴 **Critical**: App crashes, data loss, security issues
- 🟡 **High**: Major feature broken, performance issues
- 🟢 **Medium**: Minor feature issues, UI problems
- 🔵 **Low**: Documentation, cosmetic issues

---

## 💡 Feature Requests

### 📝 **Feature Request Template**

```markdown
## ✨ Feature Request

### **Feature Description**
A clear and concise description of the feature you'd like to see.

### **Problem Statement**
What problem does this feature solve? Is it related to a frustration?

### **Proposed Solution**
Describe your preferred solution or implementation approach.

### **Alternative Solutions**
Describe alternative solutions you've considered.

### **Use Cases**
Provide specific examples of how this feature would be used.

### **Implementation Complexity**
- [ ] Low (few hours)
- [ ] Medium (few days) 
- [ ] High (weeks/months)

### **Would you like to implement this?**
- [ ] Yes, I'd like to work on this
- [ ] No, I'm just suggesting
- [ ] I need help/guidance

### **Additional Context**
Add any other context, mockups, or examples.
```

---

## 🔄 Pull Request Process

### 1. **Before Submitting**
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up-to-date with main

### 2. **Pull Request Template**
```markdown
## 📋 Pull Request

### **Type of Change**
- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Refactoring (no functional changes)
- [ ] 🧪 Tests (adding or updating tests)

### **Description**
Brief description of changes and motivation.

### **Related Issues**
Fixes #(issue number)

### **Changes Made**
- List specific changes
- Include any new dependencies
- Mention configuration changes

### **Testing**
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

### **Screenshots/Demos**
If applicable, add screenshots or GIFs showing the changes.

### **Checklist**
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is commented appropriately
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] No breaking changes (or clearly documented)
```

### 3. **Review Process**
1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: At least one maintainer reviews
3. **Testing**: Manual testing for UI/UX changes
4. **Approval**: Maintainer approval required
5. **Merge**: Squash and merge preferred

### 4. **Commit Message Format**
Use **Conventional Commits** format:

```bash
# Format: type(scope): description

# Examples:
feat(model): add BERT-based emotion classifier
fix(streamlit): resolve prediction button issue
docs(readme): update installation instructions
test(api): add unit tests for prediction endpoint
refactor(utils): simplify text preprocessing pipeline
perf(inference): optimize batch prediction speed
```

**Types**:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Maintenance tasks

---

## 📊 Code Standards

### 🧹 **Code Formatting**
```bash
# Auto-format with Black
black src/ tests/ app/ --line-length 88

# Check formatting
black src/ tests/ app/ --check --diff

# Sort imports
isort src/ tests/ app/

# Lint with flake8
flake8 src/ tests/ app/ --max-line-length=88
```

### 🔧 **Pre-commit Configuration**
Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

### 📁 **Project Structure**
```
EmotiSense-AI/
├── src/                    # Source code
│   ├── models/            # Model classes
│   ├── utils/             # Utility functions  
│   ├── data/              # Data processing
│   └── visualization/     # Plotting functions
├── tests/                 # Test files
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── app/                   # Streamlit application
├── notebooks/             # Jupyter notebooks
├── outputs/               # Model outputs
├── docs/                  # Documentation
└── scripts/               # Utility scripts
```

---

## 🧪 Testing Guidelines

### 🎯 **Test Categories**

#### **Unit Tests** (`tests/unit/`)
Test individual functions and classes:

```python
# tests/unit/test_text_processor.py
import pytest
from src.utils.text_processor import TextProcessor

class TestTextProcessor:
    def test_clean_text(self):
        processor = TextProcessor()
        
        # Test basic cleaning
        result = processor.clean_text("Hello, World!!!")
        assert result == "hello world"
        
        # Test emoji handling
        result = processor.clean_text("I'm so happy! 😊")
        assert "😊" not in result
        
    def test_tokenize(self):
        processor = TextProcessor()
        
        tokens = processor.tokenize("This is a test.")
        assert isinstance(tokens, list)
        assert len(tokens) == 5
```

#### **Integration Tests** (`tests/integration/`)
Test component interactions:

```python
# tests/integration/test_model_pipeline.py
import pytest
from src.models.emotion_classifier import EmotionClassifier

class TestModelPipeline:
    def test_end_to_end_prediction(self):
        """Test complete prediction pipeline."""
        classifier = EmotionClassifier.load_pretrained(
            'outputs/emotion_classification_model'
        )
        
        # Test single prediction
        result = classifier.predict("I'm thrilled about this news!")
        
        assert result['emotion'] == 'joy'
        assert result['confidence'] > 0.8
        
    def test_batch_processing(self):
        """Test batch prediction functionality."""
        classifier = EmotionClassifier.load_pretrained(
            'outputs/emotion_classification_model'
        )
        
        texts = [
            "I love this movie!",
            "This makes me so angry!",
            "I'm scared of the dark."
        ]
        
        results = classifier.predict_batch(texts)
        
        assert len(results) == 3
        assert results[0]['emotion'] == 'love'
        assert results[1]['emotion'] == 'anger'
        assert results[2]['emotion'] == 'fear'
```

#### **Performance Tests** (`tests/performance/`)
Test speed and memory usage:

```python
# tests/performance/test_inference_speed.py
import pytest
import time
from src.models.emotion_classifier import EmotionClassifier

class TestPerformance:
    @pytest.fixture(scope="class")
    def classifier(self):
        return EmotionClassifier.load_pretrained(
            'outputs/emotion_classification_model'
        )
    
    def test_single_prediction_speed(self, classifier):
        """Single prediction should complete within reasonable time."""
        text = "This is a test sentence for speed measurement."
        
        start_time = time.time()
        result = classifier.predict(text)
        elapsed = time.time() - start_time
        
        assert elapsed < 1.0  # Should complete within 1 second
        
    def test_batch_prediction_throughput(self, classifier):
        """Batch prediction should process texts efficiently."""
        texts = ["Test sentence"] * 100
        
        start_time = time.time()
        results = classifier.predict_batch(texts, batch_size=16)
        elapsed = time.time() - start_time
        
        throughput = len(texts) / elapsed
        assert throughput > 50  # At least 50 texts per second
```

### 🏃‍♂️ **Running Tests**

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/performance/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run tests matching pattern
pytest tests/ -k "test_prediction" -v

# Run slow tests (marked with @pytest.mark.slow)
pytest tests/ -m slow

# Skip slow tests
pytest tests/ -m "not slow"

# Parallel testing (requires pytest-xdist)
pytest tests/ -n auto
```

---

## 📚 Documentation

### 📝 **Documentation Types**

#### **Code Documentation**
- **Docstrings**: All public functions/classes
- **Type hints**: Function signatures
- **Inline comments**: Complex logic explanation

#### **User Documentation**
- **README.md**: Project overview and quick start
- **API Documentation**: Function/class references
- **Tutorials**: Step-by-step guides
- **Examples**: Code snippets and notebooks

#### **Developer Documentation**
- **CONTRIBUTING.md**: This file
- **Architecture docs**: System design
- **Development guides**: Setup and workflows

### 🛠️ **Documentation Tools**

```bash
# Generate API documentation with Sphinx
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html/
```

### 📖 **Writing Guidelines**

#### **README Sections**
- Clear project description
- Installation instructions
- Quick start examples
- API reference links
- Contributing guidelines

#### **Code Comments**
```python
# Good: Explains WHY, not WHAT
# Use balanced class weights to handle dataset imbalance
class_weights = compute_class_weight('balanced', classes=unique_labels, y=y_train)

# Bad: Explains WHAT (obvious from code)
# Create a dictionary
emotion_dict = {}
```

#### **Tutorial Structure**
1. **Learning Objectives**: What will be covered
2. **Prerequisites**: Required knowledge/setup
3. **Step-by-Step Instructions**: Clear, testable steps
4. **Code Examples**: Working code snippets
5. **Troubleshooting**: Common issues and solutions
6. **Next Steps**: Links to advanced topics

---

## 🏆 Recognition

### 👏 **Contributor Recognition**
We recognize contributions in several ways:

#### **All Contributors**
We use the [All Contributors](https://allcontributors.org/) specification to recognize all types of contributions:

- 💻 **Code**: Writing code
- 📖 **Documentation**: Writing documentation
- 🐛 **Bug reports**: Reporting bugs
- 💡 **Ideas**: Contributing ideas
- 🤔 **Answering Questions**: Helping users
- 📢 **Talks**: Giving talks about the project
- 🎨 **Design**: UI/UX design
- 📊 **Data**: Contributing datasets

#### **Contributor Levels**
- **First-time contributors**: Welcome badge
- **Regular contributors**: Listed in README
- **Core contributors**: Repository access
- **Maintainers**: Full project permissions

#### **Special Recognition**
- **Monthly highlights**: Featured contributors
- **Annual awards**: Outstanding contributions
- **Conference opportunities**: Speaking at events
- **Research collaborations**: Academic partnerships

### 📈 **Contribution Tracking**
We track contributions using:
- GitHub contribution graphs
- Issue and PR participation
- Code review engagement
- Community support activities

---

## 📞 Getting Help

### 💬 **Communication Channels**

#### **GitHub Discussions**
- **General questions**: Project usage and setup
- **Feature discussions**: Brainstorming new ideas
- **Research topics**: ML and NLP discussions
- **Community showcase**: Share your projects

#### **Issue Tracker**
- **Bug reports**: Technical problems
- **Feature requests**: New functionality
- **Documentation issues**: Unclear or missing docs

#### **Direct Contact**
- **Maintainer**: [@Sharnabh](https://github.com/Sharnabh)
- **Email**: banerjeesharnabh@gmail.com
- **LinkedIn**: [linkedin.com/in/sharnabh-banerjee](https://linkedin.com/in/sharnabh-banerjee)

### 🆘 **Common Issues**

#### **Development Setup Problems**
```bash
# Clear pip cache
pip cache purge

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### **Model Loading Issues**
```bash
# Verify model files exist
ls -la outputs/emotion_classification_model/

# Test model loading
python -c "from transformers import AutoModel; AutoModel.from_pretrained('outputs/emotion_classification_model')"
```

#### **Git LFS Problems**
```bash
# Pull LFS files
git lfs pull

# Check LFS status
git lfs status

# Re-track files if needed
git lfs track "*.safetensors"
```

---

## 🎉 Thank You!

Thank you for contributing to EmotiSense-AI! Your efforts help make emotion AI more accessible and powerful for everyone. Together, we're building the future of human-computer emotional understanding.

### 🌟 **Quick Links**
- [🐙 GitHub Repository](https://github.com/Sharnabh/EmotiSense-AI)
- [📖 Documentation](https://github.com/Sharnabh/EmotiSense-AI/blob/main/README.md)
- [🐛 Report Issues](https://github.com/Sharnabh/EmotiSense-AI/issues)
- [💡 Feature Requests](https://github.com/Sharnabh/EmotiSense-AI/issues/new)
- [💬 Discussions](https://github.com/Sharnabh/EmotiSense-AI/discussions)

---

<div align="center">

**Happy Contributing! 🚀**

*Made with ❤️ by the EmotiSense-AI community*

</div>
