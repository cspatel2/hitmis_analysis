from setuptools import setup, find_packages

setup(
    name='cppython',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'tqdm>=4.60.0',
        'astropy>=4.3.0'
    ],)           