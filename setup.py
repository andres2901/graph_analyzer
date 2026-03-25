# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.	dirname(__file__))
exec(open("graph_structure/version.py").read())

setup(
    name="graph-structure",
    version=__version__, 
    description="Calculate structural properties of graphs and subgraphs based on node attributes.",
    long_description=open(path.join(here, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/andres2901/graph_structure",
    author="Andres F. Lizcano Salas",
    author_email="andres.salas-lizcano@ebc.uu.se",
    license="GNU AFFERO",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Programming Language :: Python :: 3.14",
    ],
    keywords="graph structure",
    packages=find_packages(),
    python_requires=">=3.14",
    install_requires=[
        "pandas>=3.0.1",
        "networkx>=3.6.1",
        "numpy>=1.16.5",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "graph_structure=graph_structure.graph_structure:main",
        ],
    },
    data_files=[("", ["LICENSE"])],
)

