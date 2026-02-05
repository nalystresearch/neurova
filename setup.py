# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Neurova - Advanced Image Processing and Deep Learning Library
Setup script for package installation
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# read the README file
here = Path(__file__).parent.absolute()
readme_file = here / "README.md"

long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

# read version from package
version_info = {}
version_file = here / "neurova" / "version.py"
if version_file.exists():
    with open(version_file) as f:
        exec(f.read(), version_info)
else:
    version_info["__version__"] = "0.1.0"

# core dependencies
install_requires = [
    "numpy>=1.19.0,<2.0; python_version<'3.9'",
    "numpy>=2.0; python_version>='3.9'",
]

# optional dependencies
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
    "io": [
        "pillow>=9.0.0",  # Extended image format support
    ],
    "scientific": [
        "scipy>=1.7.0",  # Advanced scientific computations
    ],
}

# all extras
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="neurova",
    version=version_info["__version__"],
    author="Squid Consultancy Group (SCG)",
    author_email="contact@nalystresearch.com",
    description="Advanced Image Processing and Deep Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nalystresearch/neurova",
    project_urls={
        "Bug Tracker": "https://github.com/nalystresearch/neurova/issues",
        "Documentation": "https://github.com/nalystresearch/neurova",
        "Source Code": "https://github.com/nalystresearch/neurova",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        "neurova": ["py.typed"],
        "neurova.data": ["**/*"],
    },
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords="image-processing computer-vision deep-learning neural-networks machine-learning",
    license="Apache-2.0",
    zip_safe=False,
)
