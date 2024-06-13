# Physics-Informed Neural Networks (PINNs) and Finite Element Method (FEM) Project. (FYS5429)

<p align="center">
  <img src="https://github.com/Odin107/FYS5429/raw/main/Project_1_and_2/figures/Diffusion/u_time_gauss.gif"  height="50%" width="50%">
  <br>
  <b>Diffusion of a 3D sine function</b>
</p>

This project focuses on the implementation and experimentation with Physics-Informed Neural Networks (PINNs) and Finite Element Method for solving differential equations related to fluid dynamics and heat diffusion. The project is divided into two main directories, Navier-Stokes and Diffusion, each targeting a specific set of problems within the domain of physics-informed learning.


## Navier-Stokes

The `PINNs_NS` directory contains code and resources for implementing PINNs to solve the Navier-Stokes equations, which describe the motion of viscous fluid substances. This part of the project explores the application of PINNs to fluid dynamics problems, aiming to model and predict fluid behavior under various conditions.


### Key Components:

- `PINNs_NS.py`: Main script for defining and training the PINN model for Navier-Stokes equations.
- `PINNs_NS_Tanh.py`: A variant of the PINN model using the Tanh activation function.
- `plot_comparison.py`: Script for plotting and comparing results from different models or configurations.
- `figures/`: Directory for storing generated plots and figures.
- `runtime_*.txt`: Files containing runtime performance metrics for different configurations.


### Usage

To run the main Navier-Stokes PINN model:

```bash
python PINNs_NS/PINNs_NS.py
```

For plotting comparisons:

```bash
python PINNs_NS/plot_comparison.py
```

## PINNs_Diffusion

The `PINNs_Diffusion` directory focuses on the application of PINNs to solve heat diffusion problems. This involves modeling the transfer of heat (or diffusion of particles) in various mediums, guided by the diffusion equation.

### Key Components:

 - `PINNs_heat.py`: Core script for defining, training, and evaluating the PINN model tailored for heat diffusion problems.
 - `figures/`: Directory for storing plots and visualizations of the model's performance and predictions.
 - `runtime_*.txt`: Performance metrics files for different activation functions used in the models.


### Usage

To run the heat diffusion PINN model:

```bash
python PINNs_Diffusion/PINNs_heat.py
```

## General Information

Both parts of the project utilize PyTorch for defining and training the neural network models. The models are designed to incorporate physical laws as part of their learning process, enabling them to make predictions that adhere to the underlying physics of the problem domain.

### Prerequisites

 - Python 3.11.8
 - PyTorch 2.2.1+cu121
 - NumPy 1.26.4
 - Matplotlib 3.8.4
   
### Installation

Ensure you have Python and pip installed. Then, install the required packages:

```bash
pip install torch numpy matplotlib
```

