from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension,CUDAExtension
import os

setup(
    name='DiffNet',
    description='Compute Hybrid Bounded Distance Fields using GPUs',
    long_description=read('README.md'),
    license='MIT',
    author='Biswajit Khara/Aditya Balu',
    author_email='{bkhara,baditya}@iastate.edu',
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['DiffNet'],
    
)