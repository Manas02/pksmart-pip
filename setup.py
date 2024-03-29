from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='pksmart',
    version='0.1.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'pksmart = pksmart.main:main',
        ],
        },
    install_requires=[
        line.strip() for line in open('requirements.txt') if line.strip()
    ],
    description="PKSmart: An Open-Source Computational Model to Predict in vivo Pharmacokinetics of Small Molecules",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
