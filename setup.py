from setuptools import setup, find_packages

setup(
    name='NCM',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'mpi4py',
        # Add more dependencies as needed
    ],
    author='Sebastian Żurek',
    author_email='sebzur@codelab.pl',
    description='Norm component matrix algorithm for fast calculation of correlation sums',
    long_description='Norm component matrix algorithm for fast calculation of correlation sums, important for calculating '
                'sample entropy or approximate entropy',
    long_description_content_type='text/markdown',
    url='https://github.com/sebzur/NCM-algorithm',
    classifiers=[
        'Programming Language :: Python :: 3',
        # Add more classifiers as needed
    ],
)