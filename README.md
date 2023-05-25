# Density-estimation-using-Affine-coupling-based-Normaling-Flows
Demonstration of Normalising Flow for Density Estimation using RealNVP model implemented from scratch

This project implements a type of normalizing flow model for density estimation. Normalizing flows are a class of generative models that learn a bijective mapping between a simple latent distribution and a complex data distribution using invertible neural networks. Affine coupling is one of the building blocks of normalizing flows that performs an affine transformation on a subset of the input variables conditioned on the other subset.

The project uses Tensorflow to implement the normalizing flow model and applies it to various synthetic datasets such as moons and rings. The project also provides a Jupyter notebook tutorial that explains the theory and implementation details of normalizing flows and affine coupling layers.

The project is based on the paper “Density estimation using Real NVP” by Laurent Dinh et al. (https://arxiv.org/abs/1605.08803) and references other related works on normalizing flows.

# Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

# Installation
To install the required dependencies for this project, run the following command:

pip install -r requirements.txt
# Usage

To view the Jupyter notebook tutorial, open the file TrainingEvaluation.ipynb in your preferred notebook viewer.

# License
This project is licensed under the MIT License.

# Contributing
If you want to contribute to this project, please follow these steps:

Fork this repository
Create a new branch for your feature or bug fix
Commit and push your changes
Create a pull request with a descriptive title and a clear explanation of your changes
# Contact
If you have any questions or feedback about this project, please feel free to contact me at <sroy22@ncsu.edu>. You can also follow me on <https://www.linkedin.com/in/somshubhra-roy-36774a143/>.
