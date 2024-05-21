from setuptools import setup, find_packages
from typing import List

# Define the file path for requirements.txt
REQUIREMENTS_FILE = 'requirements.txt'

# Define the string for -e . entry
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function returns the list of requirements from the given file.
    '''
    requirements = []
    try:
        with open(file_path, 'r') as file:
            requirements = file.readlines()
            # Remove newline characters from each requirement
            requirements = [req.strip() for req in requirements]
            # Remove the -e . entry if present
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
    return requirements

setup(
    name='MlProjectGeneral_package',
    version='0.0.1',
    description='A sample Python package',
    author='Erkan',
    author_email='erkan48592@hotmail.com',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=get_requirements(REQUIREMENTS_FILE)
 
)
