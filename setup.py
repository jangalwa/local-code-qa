#!/usr/bin/env python
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="test_project", # Replace with your own name
    version="0.0.1",
    author="Abhishek Jangalwa",
    author_email="works4j@gmail.com",
    description="A python skeleton",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jangalwa",
    packages=setuptools.find_packages(where='src', exclude=['tests', 'tests.*']),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)