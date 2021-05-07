from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension,CUDAExtension
import os

def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()

setup(
    name='DiffNet',
    description='Solve PDEs using traditional Finite Element and Finite Difference based loss functions defined using torch functions',
    long_description=read('README.md'),
    license='MIT',
    author='Biswajit Khara/Aditya Balu',
    author_email='{bkhara,baditya}@iastate.edu',
    packages=['DiffNet'],
)