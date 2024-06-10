from re import A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
import time
import os

# Ensure the 'figures' directory exists
#If you want to run this code, you need to fill in you own path, and create a folder named figures.
figures_dir = "PINNs_Diffusion/figures" 
os.makedirs(figures_dir, exist_ok=True)


def seed_everything(seed=42):
    """
    Set the random seed for reproducibility.

    Parameters:
        seed (int): The seed value to set for random number generators.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) class.

    This class defines a neural network model designed to solve physics-informed problems using deep learning.

    Attributes:
        device (torch.device): The device to which the model is moved CPU as I dont have an Nvidia grafics card.
        model (torch.nn.Sequential): The sequential model consisting of input, hidden, and output layers.

    Methods:
        forward(x):
            Defines the forward pass of the model.
    """

    def __init__(self, num_inputs, num_layers, num_neurons, device):
        """
        Initializes the PINN model with the specified architecture.

        Args:
            num_inputs (int): Number of input features.
            num_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
            device (torch.device): The device on which the model will be run (CPU or GPU).
        """

        super(PINN, self).__init__()
        self.device = device
        num_layers = num_layers
        num_neurons = num_neurons
        activation = nn.SiLU()

        layers = [nn.Linear(num_inputs, num_neurons), activation] # Input layer
        for _ in range(num_layers ):
            layers += [nn.Linear(num_neurons, num_neurons), activation] # Hidden layers
        layers += [nn.Linear(num_neurons, 1)] # Output layer

        self.model = nn.Sequential(*layers).to(device) #Create the model by stacking the layers sequentially and move to device

        # Initialize weights using Xavier uniform initialization and biases to zero
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        return self.model(x)
    

def pde(x, model, D = 1):
    """
    Defines the partial differential equation (PDE) for a physics-informed neural network.

    This function computes the residual of the PDE (2D diffusion equation), which can be used to enforce the physical constraints
    in the training of the neural network model.

    Args:
        x (torch.Tensor): Input tensor containing spatial and temporal coordinates with shape (n_samples, 3).
                          The first two columns represent spatial coordinates (x, y), and the third column represents temporal coordinates (t).
        model (torch.nn.Module): The neural network model that predicts the function u.
        D (float, optional): The diffusion coefficient. Default is 1.

    Returns:
        torch.Tensor: The residual of the PDE, which is used to enforce the physical constraints during training.
    """
    x_space = x[:, 0:1] #Spacial coordinates
    y_space = x[:, 1:2] #Spacial coordinates
    x_time = x[:, 2:3] #Temporal coordinates


    # Enable gradient tracking for x_space and x_time
    x_space.requires_grad_(True)
    y_space.requires_grad_(True)
    x_time.requires_grad_(True)


    # Predict the function u using the model
    u = model(torch.cat((x_space, y_space, x_time), dim=1))

    # Compute the first spatial derivative of u
    u_x = torch.autograd.grad(
        u, x_space, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Compute the second spatial derivative of u
    u_xx = torch.autograd.grad(
        u_x, x_space, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    u_y = torch.autograd.grad(
        u, y_space, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    u_yy = torch.autograd.grad(
        u_y, y_space, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    # Compute the temporal derivative of u
    u_t = torch.autograd.grad(
        u, x_time, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    return u_t - D * (u_xx + u_yy)


def IC(x, model):
    """
    Computes the initial condition for the PDE solution.

    Args:
        x (torch.Tensor): Input tensor containing spatial coordinates with shape (n_samples, 2).
                          The first column represents x-coordinates and the second column represents y-coordinates.
        model (torch.nn.Module): The neural network model that predicts the function u.

    Returns:
        torch.Tensor: The difference between the predicted initial condition and the true initial condition.
    """
    u0 = model(x)
    a = 5
    initial_condition = torch.exp(-a * x[:, 0:1]**2 - a * x[:, 1:2]**2)
    return u0 - initial_condition


def BC_left(x, model):
    """
    Computes the left boundary condition for the PDE solution.

    Args:
        x (torch.Tensor): Input tensor containing boundary points where the condition should be applied.
        model (torch.nn.Module): The neural network model that predicts the function u.

    Returns:
        torch.Tensor: The loss associated with enforcing the left boundary condition.
    """
    # Assuming x contains boundary points where the condition should be applied
    u_left = model(x)
    # Enforce zero boundary condition
    boundary_condition = torch.zeros_like(u_left)
    # Calculate the loss for the boundary condition
    loss = u_left - boundary_condition
    return loss


def BC_right(x, model):
    """
    Computes the right boundary condition for the PDE solution.

    Args:
        x (torch.Tensor): Input tensor containing boundary points where the condition should be applied.
        model (torch.nn.Module): The neural network model that predicts the function u.

    Returns:
        torch.Tensor: The loss associated with enforcing the right boundary condition.
    """
    # Similar approach for the right boundary
    u_right = model(x)
    boundary_condition = torch.zeros_like(u_right)
    loss = u_right - boundary_condition
    return loss


def BC_top(x, model):
    """
    Computes the top boundary condition for the PDE solution.

    Args:
        x (torch.Tensor): Input tensor containing boundary points where the condition should be applied.
        model (torch.nn.Module): The neural network model that predicts the function u.

    Returns:
        torch.Tensor: The loss associated with enforcing the top boundary condition.
    """
    # Similar approach for the top boundary
    u_top = model(x)
    boundary_condition = torch.zeros_like(u_top)
    loss = u_top - boundary_condition
    return loss


def BC_bottom(x, model):
    """
    Computes the bottom boundary condition for the PDE solution.

    Args:
        x (torch.Tensor): Input tensor containing boundary points where the condition should be applied.
        model (torch.nn.Module): The neural network model that predicts the function u.

    Returns:
        torch.Tensor: The loss associated with enforcing the bottom boundary condition.
    """
    # Similar approach for the bottom boundary
    u_bottom = model(x)
    boundary_condition = torch.zeros_like(u_bottom)
    loss = u_bottom - boundary_condition
    return loss


def train_step(X_left, X_right, X_top, X_bottom, X_collocation, X_ic, optimizer, model):
    """
    Performs a single training step for the physics-informed neural network (PINN).

    Args:
        X_left (torch.Tensor): Input tensor for the left boundary condition.
        X_right (torch.Tensor): Input tensor for the right boundary condition.
        X_top (torch.Tensor): Input tensor for the top boundary condition.
        X_bottom (torch.Tensor): Input tensor for the bottom boundary condition.
        X_collocation (torch.Tensor): Input tensor for the collocation points in the domain.
        X_ic (torch.Tensor): Input tensor for the initial condition.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        model (torch.nn.Module): The neural network model to be trained.

    Returns:
        tuple: A tuple containing the PDE loss, initial condition loss, boundary condition losses (left, right, top, bottom), and total loss.
    """
    optimizer.zero_grad() # Reset gradients

    IC_loss = torch.mean(torch.square(IC(X_ic,model)))
    BC_left_loss = torch.mean(torch.square(BC_left(X_left,model)))
    BC_right_loss = torch.mean(torch.square(BC_right(X_right,model)))
    BC_top_loss = torch.mean(torch.square(BC_top(X_top,model)))
    BC_bottom_loss = torch.mean(torch.square(BC_bottom(X_bottom,model)))
    pde_loss = torch.mean(torch.square(pde(X_collocation,model)))
    loss = IC_loss + BC_top_loss + BC_bottom_loss + BC_left_loss + BC_right_loss + pde_loss
    loss.backward() # Backpropagate loss

    
    optimizer.step() # Update weights

    return pde_loss, IC_loss, BC_left_loss, BC_right_loss, BC_top_loss, BC_bottom_loss, loss



def analytical_solution(x, t, D, a):
    """
    Computes the analytical solution for the given PDE.

    Args:
        x (np.ndarray): Input array containing spatial coordinates with shape (n_samples, 2).
                        The first column represents x-coordinates and the second column represents y-coordinates.
        t (np.ndarray): Input array containing temporal coordinates with shape (n_samples,).
        D (float): Diffusion coefficient.
        a (float): Parameter for the initial condition.

    Returns:
        np.ndarray: The analytical solution evaluated at the given spatial and temporal coordinates.
    """
    factor = a / (1 + 4 * a * D * t)
    
    # Handling t == 0 and very small t values separately
    small_t = 1e-10
    t = np.where(t == 0, small_t, t)
    
    return np.where(t == small_t,
                    np.exp(-factor * (x[:, 0:1]**2 + x[:, 1:2]**2)),
                    (1 / (4 * np.pi * D * t)) * np.exp(-(x[:, 0:1]**2 + x[:, 1:2]**2) / (4 * D * t)))


# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Generate training data
N_space = 20
N_time = 20
# Define the spatial and temporal coordinates of the training data
x_space = np.linspace(-2, 2, N_space)
y_space = np.linspace(-2, 2, N_space)
x_time = np.linspace(0, 1, N_time)

# Remove the boundary points from the collocation points (Used for evaluating the PDE)
x_collocation = x_space[1:-1]
y_collocation = y_space[1:-1]
t_collocation = x_time[1:-1]

# Meshgrid for collocation points
x_collocation, y_collocation, t_collocation = np.meshgrid(x_collocation, y_collocation, t_collocation)
x_collocation = x_collocation.reshape(-1, 1)
y_collocation = y_collocation.reshape(-1, 1)
t_collocation = t_collocation.reshape(-1, 1)

X_collocation = np.hstack((x_collocation, y_collocation, t_collocation))

# Convert the coordinates to tensors on the chosen device
X_collocation = torch.tensor(X_collocation, dtype=torch.float32, device=device)

# Define the initial condition coordinates
t_ic = np.zeros_like(x_space)
x_ic = x_space
y_ic = y_space

# Reshape the coordinates to be column vectors
x_ic = x_ic.reshape(-1, 1)
y_ic = y_ic.reshape(-1, 1)
t_ic = t_ic.reshape(-1, 1)

# Combine the spatial and temporal coordinates
X_ic = np.hstack((x_ic, y_ic, t_ic))

# Convert the coordinates to tensors on the chosen device
X_ic = torch.tensor(X_ic, dtype=torch.float32, device=device)

# Define the left and right boundary coordianates
x_left = x_space[0]*np.ones_like(x_space)
y_left = y_space[0]*np.ones_like(y_space)
t_left = x_time
x_left = x_left.reshape(-1, 1)
y_left = y_left.reshape(-1, 1)
t_left = t_left.reshape(-1, 1)
X_left = np.hstack((x_left, y_left, t_left))
X_left = torch.tensor(X_left, dtype=torch.float32, device=device)

x_right = x_space[-1]*np.ones_like(x_space)
y_right = y_space[-1]*np.ones_like(y_space)
t_right = x_time
x_right = x_right.reshape(-1, 1)
y_right = y_right.reshape(-1, 1)
t_right = t_right.reshape(-1, 1)
X_right = np.hstack((x_right, y_right, t_right))
X_right = torch.tensor(X_right, dtype=torch.float32, device=device)

# Define the top boundary coordinates
x_top = x_space * np.ones_like(y_space)
y_top = y_space[-1] * np.ones_like(x_space)
t_top = x_time
x_top = x_top.reshape(-1, 1)
y_top = y_top.reshape(-1, 1)
t_top = t_top.reshape(-1, 1)
X_top = np.hstack((x_top, y_top, t_top))
X_top = torch.tensor(X_top, dtype=torch.float32, device=device)

# Define the bottom boundary coordinates
x_bottom = x_space * np.ones_like(y_space)
y_bottom = y_space[0] * np.ones_like(x_space)
t_bottom = x_time
x_bottom = x_bottom.reshape(-1, 1)
y_bottom = y_bottom.reshape(-1, 1)
t_bottom = t_bottom.reshape(-1, 1)
X_bottom = np.hstack((x_bottom, y_bottom, t_bottom))
X_bottom = torch.tensor(X_bottom, dtype=torch.float32, device=device)


# Initialize model
model = PINN(num_inputs=3, num_layers=4, num_neurons=32, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

# Train model
epochs = 15000 # Number of training iterations
loss_list = []
epoch_list = []

# Train model
for epoch in range(epochs):
    pde_loss, IC_loss, BC_left_loss, BC_right_loss, BC_top_loss, BC_bottom_loss, loss_total = train_step(
        X_left, X_right, X_top, X_bottom, X_collocation, X_ic, optimizer,model)
    
    if epoch % 100 == 0:
        loss_list.append(loss_total.item())
        epoch_list.append(epoch)
        scheduler.step() # Update learning rate
        print(f'Epoch {epoch}, pde {pde_loss.item():.4e} , IC {IC_loss.item():.4e}, BC_left {BC_left_loss.item():.4e}, BC_right {BC_right_loss.item():.4e}, loss {loss_total.item():.4e}')


plt.plot(epoch_list, loss_list)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(figures_dir, f'Loss.png'))
plt.show()


# Generate data for plotting
N_space_plot = 100
N_time_plot = 100

x_space_plot = np.linspace(-2, 2, N_space_plot)
y_space_plot = np.linspace(-2, 2, N_space_plot)
x_time_plot = np.linspace(0, 1, N_time_plot)

x_space_mesh, y_space_mesh, x_time_mesh = np.meshgrid(x_space_plot, y_space_plot, x_time_plot)

x_space_flat = x_space_mesh.reshape(-1, 1)
y_space_flat = y_space_mesh.reshape(-1, 1)
x_time_flat = x_time_mesh.reshape(-1, 1)

x = np.hstack((x_space_flat, y_space_flat, x_time_flat))
x_tensor = torch.tensor(x, dtype=torch.float32, device=device)



# Assuming boundary_value is the value you want to set at the boundaries
boundary_value = 0

# Reshape the analytical solution
analytical = analytical_solution(x, a=5, D=1, t=x_time_flat).reshape(N_space_plot, N_space_plot, N_time_plot)

# Apply boundary conditions to the analytical solution
analytical[0, :, :] = boundary_value  # Top boundary
analytical[-1, :, :] = boundary_value  # Bottom boundary
analytical[:, 0, :] = boundary_value  # Left boundary
analytical[:, -1, :] = boundary_value  # Right boundary

# Reshape the model prediction
y_pred = model(x_tensor).detach().cpu().numpy().reshape(N_space_plot, N_space_plot, N_time_plot)

# Apply boundary conditions to the model prediction
y_pred[0, :, :] = boundary_value  # Top boundary
y_pred[-1, :, :] = boundary_value  # Bottom boundary
y_pred[:, 0, :] = boundary_value  # Left boundary
y_pred[:, -1, :] = boundary_value  # Right boundary

# Select time points to visualize, e.g., start, middle, end
time_points = [0, N_time_plot // 2, N_time_plot - 1]
for t in time_points:
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Analytical and Predicted Solution at Time = {t}')
    
    analytical_slice = analytical[:, :, t]
    y_pred_slice = y_pred[:, :, t]

    # Analytical Solution
    c0 = axs[0].contourf(analytical_slice, levels=100)
    plt.colorbar(c0, ax=axs[0])
    axs[0].set_title('Analytical')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    
    # Predicted Solution
    c1 = axs[1].contourf(y_pred_slice, levels=100)
    plt.colorbar(c1, ax=axs[1])
    axs[1].set_title('Predicted')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    
    # Absolute Error
    c2 = axs[2].contourf(np.abs(analytical_slice - y_pred_slice), levels=100, cmap='cividis')
    plt.colorbar(c2, ax=axs[2])
    axs[2].set_title('Absolute Error')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    
    # Calculate and Display RMSE for this time point
    RMSE = np.sqrt(np.mean((analytical_slice - y_pred_slice) ** 2))
    fig.suptitle(f'Time = {t}, RMSE = {RMSE:.4e}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'solution_t_{t}.png'))
    plt.show()
