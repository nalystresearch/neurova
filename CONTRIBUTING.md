# Contributing to Neurova

We appreciate your interest in contributing to Neurova. This document outlines how you can help improve the project, whether through bug reports, documentation, or code contributions.

## How You Can Contribute

### Report Issues

If you encounter a bug or unexpected behavior:

- Check existing issues to see if it's already reported
- Create a new issue with a clear title and description
- Include steps to reproduce the problem
- Provide your Python version, OS, and Neurova version
- Share relevant code snippets and error messages

### Improve Documentation

Documentation improvements are always welcome:

- Fix typos or clarify existing docs
- Add examples for undocumented features
- Write tutorials for common use cases
- Improve API documentation

### Contribute Code

Code contributions can include:

- Bug fixes
- New features (discuss in an issue first for larger changes)
- Performance improvements
- Test coverage improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/neurova.git
cd neurova
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
# Create a branch for your changes
git checkout -b your-feature-name
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

Example:

```python
def process_image(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply threshold to input image.

    Args:
        image: Input image array
        threshold: Threshold value between 0 and 1

    Returns:
        Thresholded image array

    Raises:
        ValueError: If threshold is outside valid range
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    return (image > threshold).astype(np.uint8)
```

### Testing

All code changes should include tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neurova tests/

# Run specific test file
pytest tests/test_your_feature.py
```

Test guidelines:

- Write tests for new functionality
- Ensure existing tests still pass
- Aim for high coverage of new code
- Test edge cases and error conditions

### Code Quality Tools

Before submitting, run these checks:

```bash
# Format code
black neurova/

# Check style
flake8 neurova/

# Type checking
mypy neurova/
```

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass
- [ ] Code follows project style guidelines
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)
- [ ] Commits are clear and focused

### 2. Submit Pull Request

- Push your branch to your fork
- Create a pull request against the main repository
- Fill out the PR template completely
- Link any related issues

### 3. PR Title Format

Use clear, descriptive titles:

- `Fix: Correct edge detection boundary handling`
- `Feature: Add bilateral filter implementation`
- `Docs: Update installation instructions for macOS`
- `Test: Add coverage for transform module`

### 4. Description

Provide a clear description:

- What problem does this solve?
- How does it work?
- Are there any breaking changes?
- Related issues or PRs

### 5. Review Process

- Maintainers will review your PR
- Address feedback and requested changes
- Keep discussions focused and respectful
- Be patient - reviews may take time

## Commit Message Guidelines

Write clear commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what changed and why, not how.

- Bullet points are fine
- Reference issues: Fixes #123
```

Examples:

```
Fix memory leak in GPU array transfer

Add support for TIFF image format

Improve performance of Gaussian filter by 30%
```

## Code Review Standards

Reviewers will check:

- Code correctness and functionality
- Test coverage and quality
- Documentation completeness
- Performance implications
- Compatibility with existing code
- Security considerations

## Types of Contributions

### Bug Fixes

Small fixes are always welcome:

- Fix typos
- Correct minor bugs
- Improve error messages
- Update outdated documentation

### New Features

For new features:

1. Open an issue to discuss the feature first
2. Get feedback from maintainers
3. Implement the feature
4. Add comprehensive tests
5. Update documentation

### Performance Improvements

When improving performance:

- Include benchmarks showing improvement
- Ensure correctness is maintained
- Document any trade-offs
- Test on different hardware if GPU-related

## Documentation

### Docstring Format

Use NumPy-style docstrings:

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Brief one-line description.

    More detailed description if needed. Can span multiple
    lines and include examples.

    Parameters
    ----------
    param1 : type1
        Description of param1
    param2 : type2
        Description of param2

    Returns
    -------
    return_type
        Description of return value

    Raises
    ------
    ValueError
        When param1 is invalid

    Examples
    --------
    >>> result = function_name(10, 20)
    >>> print(result)
    30
    """
```

### README and Guides

- Keep examples runnable and tested
- Use clear, simple language
- Include expected output
- Link to related documentation

## Community Guidelines

### Be Respectful

- Use welcoming and inclusive language
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Be Collaborative

- Help others when you can
- Share knowledge and insights
- Give credit where it's due
- Ask questions if unsure

### Be Professional

- Keep discussions on topic
- Avoid personal attacks
- Respect people's time
- Follow through on commitments

## Getting Help

If you need help:

- Check existing documentation
- Search closed issues
- Ask in GitHub Discussions
- Reach out to maintainers

## License

By contributing to Neurova, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:

- CHANGELOG.md for each release
- GitHub contributors page
- Project documentation

## Questions?

If anything is unclear or you need help, feel free to:

- Open an issue with your question
- Start a discussion on GitHub
- Contact the maintainers

Thank you for contributing to Neurova!
