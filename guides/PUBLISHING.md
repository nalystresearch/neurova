# Publishing Neurova to PyPI

This document outlines the steps to publish Neurova to the Python Package Index (PyPI).

## Prerequisites

Before publishing, ensure:

- [ ] All tests pass: `pytest`
- [ ] Code quality checks pass: `flake8 neurova/` and `mypy neurova/`
- [ ] Version is updated in `neurova/version.py` and `pyproject.toml`
- [ ] CHANGELOG.md is updated with release notes
- [ ] Documentation is current
- [ ] Git working directory is clean
- [ ] You have PyPI account credentials

## Setup PyPI Credentials

### 1. Create PyPI Account

- Main PyPI: https://pypi.org/account/register/
- Test PyPI (for testing): https://test.pypi.org/account/register/

### 2. Generate API Token

1. Log in to PyPI
2. Go to Account Settings
3. Create API token with scope for this project
4. Save the token securely

### 3. Configure credentials

Create or edit `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
```

## Build Process

### 1. Clean Previous Builds

```bash
# Remove old builds
rm -rf dist/ build/ *.egg-info neurova.egg-info
```

### 2. Install Build Tools

```bash
# Upgrade pip and install build tools
python -m pip install --upgrade pip
pip install --upgrade build twine
```

### 3. Update Version

Edit `neurova/version.py`:

```python
__version__ = "0.1.0"  # Update version number
__version_info__ = (0, 1, 0)  # Update version tuple
__date__ = "2026-02-03"  # Update date
```

And `pyproject.toml`:

```toml
[project]
version = "0.1.0"  # Must match version.py
```

### 4. Build Distribution

```bash
# Build source distribution and wheel
python -m build

# This creates:
# - dist/neurova-0.1.0.tar.gz (source distribution)
# - dist/neurova-0.1.0-py3-none-any.whl (wheel)
```

### 5. Verify Build

```bash
# Check package contents
tar -tzf dist/neurova-0.1.0.tar.gz | head -20
unzip -l dist/neurova-0.1.0-py3-none-any.whl | head -20

# Verify metadata
twine check dist/*
```

### 6. Test Installation Locally

```bash
# Create test environment
python -m venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate

# Install from wheel
pip install dist/neurova-0.1.0-py3-none-any.whl

# Test import
python -c "import neurova; print(neurova.__version__)"

# Deactivate and remove test environment
deactivate
rm -rf test-env
```

## Publishing

### Test PyPI (Recommended First)

1. **Upload to Test PyPI:**

```bash
python -m twine upload --repository testpypi dist/*
```

2. **Test installation from Test PyPI:**

```bash
# Create fresh environment
python -m venv test-env
source test-env/bin/activate

# Install from Test PyPI (may need to specify dependencies from main PyPI)
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    neurova

# Verify
python -c "import neurova; print(neurova.__version__)"

# Clean up
deactivate
rm -rf test-env
```

### Production PyPI

Once testing is successful:

```bash
# Upload to PyPI
python -m twine upload dist/*

# You'll be prompted for credentials (or use API token)
```

## Post-Release

### 1. Tag Release in Git

```bash
# Create annotated tag
git tag -a v0.1.0 -m "Release version 0.1.0"

# Push tag to GitHub
git push origin v0.1.0
```

### 2. Create GitHub Release

1. Go to https://github.com/neurova/neurova/releases
2. Click "Create a new release"
3. Select the tag you just created
4. Title: "v0.1.0"
5. Description: Copy from CHANGELOG.md
6. Attach dist files if desired
7. Publish release

### 3. Verify Package on PyPI

- Check https://pypi.org/project/neurova/
- Verify README renders correctly
- Check metadata and links
- Test installation: `pip install neurova`

### 4. Update Documentation

- Update installation instructions if needed
- Announce release in README
- Update any version-specific documentation

## Troubleshooting

### Build Fails

```bash
# Check setup.py and pyproject.toml syntax
python setup.py check

# Ensure all required files exist
ls README.md LICENSE pyproject.toml setup.py
```

### Upload Fails

```bash
# Check credentials
cat ~/.pypirc

# Verify package name isn't taken
pip search neurova  # (if pip search is available)
# Or check https://pypi.org/project/neurova/

# Check for API token issues
python -m twine upload --verbose dist/*
```

### Version Conflicts

```bash
# Ensure version is consistent
grep version neurova/version.py pyproject.toml setup.py

# PyPI won't allow re-uploading same version
# Must bump version for new upload
```

## Version Bumping Guidelines

### Semantic Versioning

- **Major (X.0.0)**: Breaking changes, incompatible API changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

### Pre-release Versions

- Alpha: `0.1.0a1` (very unstable)
- Beta: `0.1.0b1` (feature complete, needs testing)
- Release Candidate: `0.1.0rc1` (nearly ready)
- Final: `0.1.0`

## Quick Reference

```bash
# Complete release workflow
rm -rf dist/ build/ *.egg-info
python -m pip install --upgrade pip build twine
python -m build
twine check dist/*
python -m twine upload --repository testpypi dist/*  # Test first
python -m twine upload dist/*  # Production
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

## Security Considerations

- Never commit PyPI tokens to version control
- Use API tokens instead of passwords
- Enable 2FA on your PyPI account
- Regularly rotate API tokens
- Keep `~/.pypirc` file permissions secure: `chmod 600 ~/.pypirc`

## Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PyPI Help](https://pypi.org/help/)
- [Semantic Versioning](https://semver.org/)

## Support

If you encounter issues during publishing:

1. Check PyPI status: https://status.python.org/
2. Review Python Packaging User Guide
3. Contact PyPI support if needed
