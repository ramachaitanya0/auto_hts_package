from setuptools import find_packages, setup
import pathlib
import os


setup(
    name = "Auto_hts",
    version = "0.0.1",
    author = "Ishita Roy,  Rama Chaitanya Karanam",
    author_email = "ramachaitanya0@gmail.com",
    description = 'AUTO HIERARCHIAL TIME SERIES FORECASTING',
    # long_description = readme,
    long_description_content_type = 'text/markdown',
    url = "https://github.com/ramachaitanya0/auto_hts_package",

    classifiers = [ 
        "Programming Language :: Python :: 3 ",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ] ,
    

    license='MIT',
    packages = find_packages(),
)