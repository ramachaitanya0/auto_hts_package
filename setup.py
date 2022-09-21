from setuptools import find_packages, setup
import pathlib
import os

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name = "auto_hts",
    version = "0.0.1",
    author = "Ishita Roy,  Rama Chaitanya Karanam",
    author_email = "ramachaitanya0@gmail.com",
    description = 'AUTO HIERARCHIAL TIME SERIES FORECASTING',
    long_description = README,
    long_description_content_type = 'text/markdown',
    packages = find_packages(),
    license="MIT",
    install_requires=["scikit_learn==1.0.2","scikit-hts==0.5.12","arch==5.2.0","pmdarima==1.8.5","fbprophet==0.7.1"],
    url = "https://github.com/ramachaitanya0/auto_hts_package",

    classifiers = [ 
        "Programming Language :: Python :: 3 ",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ] 
       
    
)