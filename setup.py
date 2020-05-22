from setuptools import find_packages
from setuptools import setup

setup(
    name='gcp-ml',
    packages=find_packages(),
    install_requires=[
        "tensorflow-io",
        "google-cloud-bigquery-storage",
        "grpcio"
    ]
)
