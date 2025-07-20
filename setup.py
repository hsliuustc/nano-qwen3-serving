#!/usr/bin/env python3
"""
Setup script for nano-qwen3-serving.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nano-qwen3-serving",
    version="0.1.0",
    author="Your Name",
    author_email="your-email@example.com",
    description="A high-performance, OpenAI-compatible API server for Qwen3 models on Apple Silicon",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nano-qwen3-serving",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/nano-qwen3-serving/issues",
        "Source": "https://github.com/yourusername/nano-qwen3-serving",
        "Documentation": "https://github.com/yourusername/nano-qwen3-serving#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Framework :: FastAPI",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nano-qwen3-serving=tools.start_service:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nano_qwen3_serving": ["py.typed"],
    },
    zip_safe=False,
    keywords=[
        "llm",
        "qwen3",
        "openai",
        "api",
        "apple-silicon",
        "mps",
        "fastapi",
        "streaming",
        "inference",
        "ai",
        "machine-learning",
    ],
) 