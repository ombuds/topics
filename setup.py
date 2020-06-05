import os
from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Document classification model and service built using the Yahoo! Question-Answer dataset.',
    author='Thomas Reid',
    license='',
    # TODO: add entry points to simplify module interaction
    # entry_points={
    #     'console_scripts': [
    #         'topics = src.main:main',
    #     ],
    # },
    install_requires=['spacy','click','flask','pandas','pip','pylint','pytest']
)
