"""
LearningByDoing NeurIPS 2021 Competition:
standalone code and results
"""


import io
import re
from pathlib import Path
from setuptools import setup


requirements = [
    'control==0.9.0',
    'cvxpy==1.1.13',
    'GPy==1.10.0',
    'matplotlib==3.3.4',
    'numpy==1.19.5',
    'onnxruntime==1.8.0',
    'pandas==1.2.1',
    'pyzmq==22.1.0',
    'scikit-learn==0.24.2',
    'skl2onnx==1.8.0',
    'scipy==1.6.0',
    'faiss-cpu==1.7.1.post2',
    'tqdm',
    'pytest',
]


lbd_package_name = "lbd-comp"
lbd_package_dir_name = "lbd_comp"


# Based on https://github.com/albumentations-team/albumentations/blob/master/setup.py
def package_version():
    current_dir = Path(__file__).parent.absolute()
    version_file_path = current_dir.joinpath(
        lbd_package_dir_name,
        "__init__.py",
    )
    with io.open(version_file_path, encoding="utf-8") as f:
        pkg_version = re.search(
            r'^__version__ = [\'"]([^\'"]*)[\'"]',
            f.read(),
            re.M,
        ).group(1)
        return pkg_version


setup(
    name=lbd_package_name,
    version=package_version(),
    author="Learning By Doing",
    author_email="LearningByDoing@math.ku.dk",
    maintainer_email="tel1@andrew.cmu.edu",
    package_dir = {"": "."},
    packages=[lbd_package_dir_name],
    install_requires=requirements
)
