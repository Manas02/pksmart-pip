from setuptools import setup, find_packages

setup(
    name='pksmart',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'pksmart = pksmart.main:main',
        ],
        },
    install_requires=[
        line.strip() for line in open('requirements.txt') if line.strip()
    ],
)
