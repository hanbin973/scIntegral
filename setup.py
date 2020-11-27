from setuptools import setup
import setuptools

setup(
		name = 'scintegral',
		version = '0.1.3',
		author = 'Hanbin Lee',
		author_email = 'hanbin973@gmail.com',
		description = 'Semi-supervised scRNA-seq cell clasifier',
		url = 'https://github.com/hanbin973/scIntegral',
		packages=setuptools.find_packages(),
		install_requires = ['numpy>=1.14', 'pandas>=1.1.0', 'torch>=1.6'],
		classifiers=[
			"Programming Language :: Python :: 3",
			"Operating System :: OS Independent",
			"Intended Audience :: Science/Research",
			"Topic :: Scientific/Engineering",
			"Topic :: Scientific/Engineering :: Bio-Informatics"
		],
		python_requires = '>=3.4',
		)
