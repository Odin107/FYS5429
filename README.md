# Going with the Flow: A Comparative Study of Physics-Informed Neural Networks and FEM for Diffusion and Navier-Stokes Equations in Poiseuille Flow. (FYS5429)

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

### Setting Up Jupyter Notebook with Docker

To set up a Jupyter Notebook environment using Docker for running the PhenicsX code, follow these steps:

1. **Install Docker**: Ensure Docker is installed on your system. You can download and install Docker from [here](https://www.docker.com/get-started).

2. **Create a Dockerfile**: Create a `Dockerfile` with the necessary configurations to set up the environment.

    ```Dockerfile
    # Use the official FEniCSx Docker image
    FROM dolfinx/dolfinx:stable

    # Install additional dependencies
    RUN pip install notebook torch numpy matplotlib

    # Set up working directory
    WORKDIR /home/fenicsx

    # Copy project files into the container
    COPY . /home/fenicsx

    # Expose port for Jupyter Notebook
    EXPOSE 8888

    # Start Jupyter Notebook
    CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
    ```

3. **Build the Docker Image**: Navigate to the directory containing the `Dockerfile` and build the Docker image.

    ```bash
    docker build -t fenicsx-notebook .
    ```

4. **Run the Docker Container**: Start a Docker container with the built image.

    ```bash
    docker run -p 8888:8888 fenicsx-notebook
    ```

5. **Access Jupyter Notebook**: Open your web browser and navigate to `http://localhost:8888`. Use the token provided in the terminal to access the Jupyter Notebook.

This is the setup used for this project as it ensures a consistent environment for running the PhenicsX code, leveraging Docker to manage dependencies and environment configurations.


