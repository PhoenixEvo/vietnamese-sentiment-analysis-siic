"""
Setup script for SIIC - Vietnamese Emotion Detection System
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="siic",
    version="1.0.0",
    author="Team InsideOut",
    author_email="team@insideout.com",
    description="Vietnamese Emotion Detection System using PhoBERT, LSTM, and traditional ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/team-insideout/siic",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "gpu": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "siic-train=siic.training.trainers:main",
            "siic-evaluate=scripts.evaluate:main", 
            "siic-dashboard=scripts.dashboard:main",
        ],
    },
    include_package_data=True,
    package_data={
        "siic": ["configs/*.yaml", "configs/*.json"],
    },
    keywords="emotion detection, vietnamese nlp, phobert, lstm, sentiment analysis",
    project_urls={
        "Bug Reports": "https://github.com/team-insideout/siic/issues",
        "Source": "https://github.com/team-insideout/siic",
        "Documentation": "https://siic.readthedocs.io/",
    },
) 