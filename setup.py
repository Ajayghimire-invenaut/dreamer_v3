from setuptools import setup, find_packages

setup(
    name="dreamer_v3",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "gym>=0.26.0",
        "ruamel.yaml>=0.17.21",
        "tensorboard>=2.10.0",
        "numpy>=1.23.5"
    ],
)
