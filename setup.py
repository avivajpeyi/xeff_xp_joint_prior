from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Effective Spin Priors'
LONG_DESCRIPTION = 'Helps calculate p(xp, xeff)'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="effective_spins",
    version=VERSION,
    author="Avi Vajpeyi",
    author_email="avi.vajpyi@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords=['prior'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)