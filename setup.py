from pathlib import Path
import re

import setuptools


ROOT = Path(__file__).parent
VERSION_FILE = ROOT / "hdlib" / "__init__.py"

INSTALL_REQUIRES = [
    "mthree>=3.0.0",
    "numpy>=2.3.4",
    "qiskit>=2.2.1",
    "qiskit-aer>=0.17.2",
    "qiskit-ibm-runtime>=0.42.0",
    "qiskit_machine_learning>=0.8.4",
    "scikit-learn>=1.7.2",
    "scipy>=1.16.2",
    "tabulate>=0.9.0",
]

TEST_REQUIRES = [
    "pytest>=8",
]

version_match = re.search(r'__version__ = "([^"]+)"', VERSION_FILE.read_text(encoding="utf-8"))
if not version_match:
    raise RuntimeError("Unable to find hdlib version string.")

setuptools.setup(
    author="Fabio Cumbo",
    author_email="fabio.cumbo@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
    ],
    description="Hyperdimensional Computing Library for building Vector Symbolic Architectures in Python",
    extras_require={
        "dev": [
            "build>=1",
            "twine>=5",
            *TEST_REQUIRES,
        ],
        "test": TEST_REQUIRES,
    },
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    license_files=["LICENSE"],
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    name="hdlib",
    packages=setuptools.find_packages(include=["hdlib", "hdlib.*"], exclude=["test", "test.*"]),
    project_urls={
        "Issues": "https://github.com/cumbof/hdlib/issues",
        "Source": "https://github.com/cumbof/hdlib",
        "Wiki": "https://github.com/cumbof/hdlib/wiki",
    },
    python_requires=">=3.11",
    scripts=[
        "examples/chopin2/chopin2.py",
    ],
    url="http://github.com/cumbof/hdlib",
    version=version_match.group(1),
    zip_safe=False,
)
