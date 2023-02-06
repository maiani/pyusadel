# pyUsadel
A toolkit for modeling superconductor heterostructure with Usadel equations written ain python.
This is code is still in a preliminary stage and work-in-progress.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Current limitations](#current)
- [Contributing](#contributing)
- [License](#license)

## Description
pyUsadel is a python library for modeling superconductor heterostructures using Usadel equations. It provides a flexible and user-friendly interface for simulating and analyzing the behavior of these systems.

## Installation
To install pyUsadel, you will need to have the following dependencies installed:
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [numba](https://numba.pydata.org/) (optional)
To install pyUsadel, run the following command:
`pip install pyusadel`

## Usage
To use pyUsadel, simply import the library in your Python script and you can access the functions and classes provided by pyUsadel to build and run your simulations. Some example simulations are included in under the folder `doc/examples/.`

## Current limitations
 - Only magnetization in the x-y plane is supported.
 - No orbital effects.
You are suggested to look at the [TODO](TODO.md) file to check for features that are still missing. 
 
## Contributing
We welcome contributions to pyUsadel! If you would like to contribute, please follow these steps:
1. Fork the repository
2. Create a new branch for your changes
3. Make your changes and test them thoroughly
4. Submit a pull request to the main repository
If you find a bug, please report it using the issue tracker.

A [TODO](TODO.md) list of task is included.

## License
pyUsadel is released under the MIT License. See [LICENSE](LICENSE) for more information.

## Authors
pyUsadel was developed by 
- Andrea Maiani (andrea.maiani@outlook.com)
