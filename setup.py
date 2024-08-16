import os
import setuptools
import sys

from hdlib import __version__

if sys.version_info[0] < 3:
    sys.stdout.write("hdlib requires Python 3 or higher. Your Python your current Python version is {}.{}.{}"
                     .format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))

REQUIREMENTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

if not os.path.isfile(REQUIREMENTS):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), REQUIREMENTS)

setuptools.setup(
    author="Fabio Cumbo",
    author_email="fabio.cumbo@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    description="Hyperdimensional Computing Library for building Vector Symbolic Architectures in Python",
    install_requires=[
        requirement.strip() for requirement in open(REQUIREMENTS).readlines() if requirement.strip()
    ],
    license="MIT",
    license_files=["LICENSE"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    name="hdlib",
    packages=setuptools.find_packages(include=["hdlib"], exclude=["test"]),
    project_urls={
        "Issues": "https://github.com/cumbof/hdlib/issues",
        "Source": "https://github.com/cumbof/hdlib",
        "Wiki": "https://github.com/cumbof/hdlib/wiki",
    },
    python_requires=">=3",
    scripts=[
        "examples/chopin2.py",
    ],
    url="http://github.com/cumbof/hdlib",
    version=__version__,
    zip_safe=False
)
