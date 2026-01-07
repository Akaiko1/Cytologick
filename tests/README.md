# Cytologick PyTorch Tests

This directory contains comprehensive tests for the PyTorch functionality in Cytologick, ensuring compatibility and correctness alongside the existing TensorFlow implementation.

## Test Structure

### Core Test Files

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`test_pytorch_training.py`** - PyTorch training pipeline tests
- **`test_pytorch_inference.py`** - PyTorch inference pipeline tests
- **`test_framework_selection.py`** - Framework switching and configuration tests
- **`test_integration.py`** - End-to-end integration tests

### Test Categories

#### 1. Training Pipeline Tests (`test_pytorch_training.py`)
- Dataset creation and loading
- Data augmentation transforms
- Loss function implementations (Lovasz + CrossEntropy)
- Metrics calculation (IoU, F1 score)
- Model architecture validation
- Training loop functionality

#### 2. Inference Pipeline Tests (`test_pytorch_inference.py`)
- Image preprocessing for PyTorch
- Model loading and saving
- Sliding window inference
- Raw probability output
- Memory efficiency
- Output format / shape compatibility

#### 3. Framework Selection Tests (`test_framework_selection.py`)
- Configuration switching between TensorFlow/PyTorch
- YAML config loading
- Case-insensitive framework selection
- Module import behavior
- Training script framework routing

#### 5. Integration Tests (`test_integration.py`)
- End-to-end pipeline testing
- Pipeline-level sanity checks
- Complete workflow validation

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r tests/requirements_test.txt
```

### Running All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=clogic

# Run specific test categories
pytest -m pytorch          # PyTorch-specific tests
pytest -m integration      # Integration tests
```

### Running Specific Test Files

```bash
# Training tests
pytest tests/test_pytorch_training.py

# Inference tests
pytest tests/test_pytorch_inference.py

# Framework selection tests
pytest tests/test_framework_selection.py

# Integration tests
pytest tests/test_integration.py
```

### Running Tests with Different Configurations

```bash
# Test PyTorch functionality only
pytest -m pytorch

# Test with verbose output
pytest -v

# Test with specific framework
CYTOLOGICK_FRAMEWORK=pytorch pytest

# Run fast tests only (exclude slow tests)
pytest -m "not slow"
```

## Test Fixtures

### Available Fixtures (`conftest.py`)

- **`temp_dir`** - Temporary directory for test files
- **`sample_image`** - Sample RGB image for testing
- **`sample_mask`** - Sample segmentation mask
- **`sample_dataset_files`** - Complete dataset structure for testing
- **`mock_config`** - Mock configuration object
- **`pytorch_device`** - Available PyTorch device (CPU/GPU)
- **`skip_if_no_pytorch`** - Skip tests if PyTorch dependencies missing
- **`original_config`** - Backup/restore original config

## Test Markers

Tests are marked with the following pytest markers:

- **`@pytest.mark.pytorch`** - PyTorch-specific functionality
- **`@pytest.mark.tensorflow`** - TensorFlow-specific functionality  
- **`@pytest.mark.slow`** - Tests that take longer to run
- **`@pytest.mark.integration`** - End-to-end integration tests

## Dependencies

### Required for Testing
- pytest >= 7.0.0
- torch >= 1.12.0
- segmentation-models-pytorch >= 0.3.0
- albumentations >= 1.3.0
- numpy, opencv-python, Pillow

### Optional for Full Testing
- tensorflow >= 2.10.0 (for compatibility tests)

## Coverage Requirements

Tests maintain minimum 70% code coverage for the `clogic` module. Coverage reports are generated in HTML format and shown in terminal output.

## Continuous Integration

Tests are designed to run in CI environments with:
- Automatic PyTorch dependency detection
- GPU/CPU compatibility
- Mocking for heavyweight dependencies where needed
- Timeout protection for long-running tests

## Troubleshooting

### Common Issues

1. **PyTorch not available**: Tests will be skipped automatically
2. **Optional deps**: install extras as needed (e.g. TensorFlow)
3. **Memory issues**: Tests include cleanup and memory management
4. **Model loading errors**: Check file paths and model compatibility

### Debug Mode

Run tests with additional debugging:
```bash
# Verbose output with full traceback
pytest -vvv --tb=long

# Stop on first failure
pytest -x

# Run specific test with debugging
pytest tests/test_pytorch_training.py::TestPyTorchTraining::test_model_architecture -vvv
```

## Contributing

When adding new PyTorch functionality:

1. Add corresponding tests in appropriate test file
2. Use existing fixtures when possible
3. Mark tests with appropriate markers
4. Ensure compatibility with both CPU and GPU
5. Include error handling tests
6. Maintain coverage requirements

### Test Naming Convention

- Test files: `test_<module>_<framework>.py`
- Test classes: `Test<Functionality>`
- Test methods: `test_<specific_behavior>`

Example:
```python
class TestPyTorchTraining:
    def test_dataset_creation(self):
        # Test dataset creation functionality
        pass
    
    def test_model_architecture(self):
        # Test model architecture validation
        pass
```