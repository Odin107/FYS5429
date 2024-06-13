# Physics-Informed Neural Networks (PINNs) and Finite Element Method (FEM) Project. (FYS5429)

<p align="center">
  <img src="https://github.com/Odin107/FYS5429/raw/main/Project_1_and_2/figures/Diffusion/u_time_gauss.gif"  height="50%" width="50%">
  <br>
  <b>Diffusion of a Gaussian hill</b>
</p>

This project focuses on the implementation and experimentation with Physics-Informed Neural Networks (PINNs) and Finite Element Method for solving differential equations related to fluid dynamics and heat diffusion. The project is divided into two main directories, Navier-Stokes and Diffusion, each targeting a specific set of problems within the domain of physics-informed learning.


## Navier-Stokes

The `Navier-Stokes` directory contains code and resources for implementing PINNs to solve the Navier-Stokes equations, which describe the motion of viscous fluid substances. This part of the project explores the application of PINNs to fluid dynamics problems, aiming to model and predict fluid behavior under various conditions.


### Key Components:

- `PINNs_NS.py`: Main script for defining and training the PINN model for Navier-Stokes equations.
- `anal_sol.py`: Script for computing analytical solutions to the Navier-Stokes equations.
- `NS_FEM.py`: Finite Element Method implementation for solving Navier-Stokes equations.


### Usage

To run the main Navier-Stokes PINN model:

```bash
python Path/to/python/file
```

## PINNs_Diffusion

The `Diffusion` directory focuses on the application of PINNs to solve the diffusion equation. This involves modeling the diffusion of a gaussian hill.

### Key Components:

 - `PINNs_heat.py`: Core script for defining, training, and evaluating the PINN model tailored for heat diffusion problems.
 - `anal_anim.py`: Script for generating animations from the analytical solutions.
 - `FEM_heat.py`: Finite Element Method (FEM) implementation for heat diffusion.


### Usage

To run the heat diffusion PINN model:

```bash
python Path/to/python/file
```

## General Information

The first part of the project utilizes PyTorch for defining and training the neural network models. The secound uses PhenicsX and DolfinX. The models are designed to incorporate physical laws as part of their learning process, enabling them to make predictions that adhere to the underlying physics of the problem domain.

### Prerequisites

 - Python 3.11.8
 - PyTorch 2.2.1+cu121
 - NumPy 1.26.4
 - Matplotlib 3.8.4
 - DolfinX 0.8.0
 - PhenicsX 0.8.0
   
### Installation

Ensure you have Python and pip installed. Then, install the required packages:

```bash
pip install torch numpy matplotlib
```
```bash
conda create -n fenicsx-env
conda activate fenicsx-env
conda install -c conda-forge fenics-dolfinx mpich pyvista
```

