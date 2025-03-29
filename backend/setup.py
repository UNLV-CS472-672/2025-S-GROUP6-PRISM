"""Created by Daniel Levy, 3/18/2025.

Python imports... suck. And the documentation sucks
even more... :)

I searched the Internet, and I found this solution.

Run the command 'pip install -e .' in the django virtual
environment, and it will install all the required dependencies.
"""

from setuptools import setup, find_packages

setup(name="prism_backend", version="1.0", packages=find_packages())
