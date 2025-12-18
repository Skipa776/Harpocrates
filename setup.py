from pathlib import Path

from setuptools import find_packages, setup

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

version = {}
version_file = Path(__file__).parent / "Harpocrates" / "__init__.py"
with open(version_file) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

setup(
    name="harpocrates",
    version=version.get("__version__", "0.1.0"),
    author="Joshua Do",
    author_email="joshalfonsodo@gmail.com",
    description="ML-powered secrets detection tool for code repositories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Skipa776/Harpocrates",
    packages=find_packages(exclude=["test",'tests.*','examples']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Security",
        "Topic :: Software Development :: Quality Assurance",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.6.0",
            "mypy>=1.0.0",
        ],
        "ml": [
            # Will add later: xgboost, scikit-learn, etc.
        ],
    },
    entry_points={
        "console_scripts": [
            "harpocrates=Harpocrates.cli:main",
        ],
    },
    include_package_data=True,
)
