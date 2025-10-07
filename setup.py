"""
PyBDR: Set-boundary based Reachability Analysis Toolbox in Python

This setup.py file is provided for backward compatibility with older Python
packaging tools. Modern installations should use pyproject.toml with pip >= 21.0.
"""

import os

from setuptools import find_packages, setup


# Read the README file for the long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Define requirements directly (use environment.yml for conda installation)
def read_requirements():
    return [
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pypoman>=0.5.4",
        "cvxpy>=1.1.0",
        "matplotlib>=3.3.0",
        "codac>=2.0.0.dev0",  # Allows pre-release versions
    ]


setup(
    name="pybdr",
    version="1.0.4",
    author="ASAG-ISCAS",
    description="Set-boundary based Reachability Analysis Toolbox in Python",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ASAG-ISCAS/PyBDR",
    project_urls={
        "Documentation": "https://asag-iscas.github.io/docs.pybdr/",
        "Source": "https://github.com/ASAG-ISCAS/PyBDR",
        "Issues": "https://github.com/ASAG-ISCAS/PyBDR/issues",
    },
    packages=find_packages(exclude=["test*", "benchmarks*", "doc*"]),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="reachability analysis, formal verification, nonlinear systems, boundary analysis, set-based methods",
    include_package_data=True,
    zip_safe=False,
)
