import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from torch.utils.data import DataLoader, TensorDataset
import random
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    """
    Set the random seed for reproducibility across various libraries and environments.

    Parameters:
    seed (int): The seed value to set for the random number generators. Default is 42.

    This function sets the seed for the following:
    - Python's built-in random module
    - NumPy
    - PyTorch
    - The PYTHONHASHSEED environment variable

    Example:
        seed_everything(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(42)

# Create a directory to store the figures
figures_dir = "PINNs_NS/figures"
os.makedirs(figures_dir, exist_ok=True)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) class for solving PDEs.

    Attributes:
    device (torch.device): The device to run the model on (CPU or GPU).
    model (nn.Sequential): The sequential model consisting of linear layers and activation functions.

    Methods:
    forward(x):
        Forward pass through the neural network.
    """
    def __init__(self, num_inputs, num_layers, num_neurons, activation, device):
        """
        Initialize the PINN model.

        Parameters:
        num_inputs (int): Number of input features.
        num_layers (int): Number of hidden layers.
        num_neurons (int): Number of neurons per hidden layer.
        activation (nn.Module): Activation function to use between layers.
        device (torch.device): Device to run the model on (CPU or GPU).
        """
        super(PINN, self).__init__()
        self.device = device

        layers = [nn.Linear(num_inputs, num_neurons), activation]
        for _ in range(num_layers):
            layers += [nn.Linear(num_neurons, num_neurons), activation]
        layers += [nn.Linear(num_neurons, 3)]

        self.model = nn.Sequential(*layers).to(device)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


    def forward(self, x):
        """
        Forward pass through the neural network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output of the neural network.
        """
        return self.model(x)


def pde(x, model):
    """
    Compute the residuals of the Navier-Stokes equations using a Physics-Informed Neural Network (PINN).

    Parameters:
    x (torch.Tensor): Input tensor of shape (N, 3) where N is the number of samples. The tensor contains
                      spatial coordinates (x_space, y_space) and temporal coordinate (x_time).
    model (nn.Module): The PINN model which outputs the predicted values of u, v, and p.

    Returns:
    tuple: A tuple containing the residuals of the Navier-Stokes equations:
        - f_u (torch.Tensor): Residual of the u-momentum equation.
        - f_v (torch.Tensor): Residual of the v-momentum equation.
        - f_c (torch.Tensor): Residual of the continuity equation.

    This function performs the following steps:
    1. Extract spatial and temporal components from the input tensor.
    2. Enable gradient tracking for these components.
    3. Use the model to predict the values of u, v, and p.
    4. Compute first and second-order derivatives of u and v with respect to spatial components.
    5. Compute first-order derivatives of u and v with respect to the temporal component.
    6. Compute first-order derivatives of p with respect to spatial components.
    7. Calculate the residuals of the Navier-Stokes equations using the computed derivatives and a predefined Reynolds number (Re).

    Example:
        f_u, f_v, f_c = pde(x, model)
    """
    x_space = x[:, 0:1]
    y_space = x[:, 1:2]
    x_time = x[:, 2:3]

    x_space.requires_grad_(True)
    y_space.requires_grad_(True)
    x_time.requires_grad_(True)

    u_v_p = model(torch.cat((x_space, y_space, x_time), dim=1))
    u, v, p = torch.split(u_v_p, 1, dim=1)

    u_x = torch.autograd.grad(u, x_space, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_space, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x_space, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y_space, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x_space, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_space, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x_space, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y_space, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    u_t = torch.autograd.grad(u, x_time, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, x_time, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x_space, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y_space, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    Re = 100
    f_u = u_t + u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy)
    f_c = u_x + v_y

    return f_u, f_v, f_c


# Define U_max and H
U_max = 1.0  # Maximum velocity at the center of the channel
H = 1.0  # Height of the channel, set to 1 as the domain is normalized


def boundary_conditions(model, x, y):
    """
    Define the boundary conditions for the Physics-Informed Neural Network (PINN).

    Parameters:
    model (nn.Module): The PINN model which outputs the predicted values of u, v, and p.
    x (torch.Tensor): Input tensor for the spatial x-coordinate.
    y (torch.Tensor): Input tensor for the spatial y-coordinate.

    Returns:
    tuple: A tuple containing the boundary conditions for the inlet, outlet, and walls:
        - bc_inlet (torch.Tensor): Boundary condition at the inlet (x = 0).
        - bc_outlet (torch.Tensor): Boundary condition at the outlet (x = 1).
        - bc_walls (torch.Tensor): Boundary condition at the walls (y = 0 and y = H).

    This function performs the following steps:
    1. Enable gradient tracking for x and y.
    2. Use the model to predict the values of u, v, and p at the given coordinates.
    3. Compute the inlet boundary condition for u.
    4. Set the outlet boundary condition for p.
    5. Define the wall boundary conditions for u and v.

    Example:
        bc_inlet, bc_outlet, bc_walls = boundary_conditions(model, x, y)
    """
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    u_v_p = model(torch.cat([x, y, torch.zeros_like(x)], dim=1))
    u, v, p = u_v_p[:, 0:1], u_v_p[:, 1:2], u_v_p[:, 2:3]

    # Inlet (x = 0)
    u_inlet = 4 * U_max * (y * (H - y) / H**2)
    bc_inlet = u - u_inlet

    # Outlet (x = 1)
    
    p_outlet = torch.zeros_like(p)
   
    bc_outlet = p - p_outlet

    # Walls (y = 0 and y = H)
    bc_walls = torch.cat([u, v], dim=1)

    return bc_inlet, bc_outlet, bc_walls


def initial_conditions(model, x, y):
    """
    Define the initial conditions for the Physics-Informed Neural Network (PINN).

    Parameters:
    model (nn.Module): The PINN model which outputs the predicted values of u, v, and p.
    x (torch.Tensor): Input tensor for the spatial x-coordinate.
    y (torch.Tensor): Input tensor for the spatial y-coordinate.

    Returns:
    tuple: A tuple containing the initial conditions for u, v, and p:
        - u_initial (torch.Tensor): Initial condition for u (should be zero).
        - v_initial (torch.Tensor): Initial condition for v (should be zero).
        - p_initial (torch.Tensor): Initial condition for p (should be zero).

    This function performs the following steps:
    1. Set the temporal coordinate t to zero.
    2. Use the model to predict the values of u, v, and p at the given coordinates.
    3. Return the initial conditions as the difference from zero.

    Example:
        u_initial, v_initial, p_initial = initial_conditions(model, x, y)
    """
    t = torch.zeros_like(x)
    u_v_p = model(torch.cat([x, y, t], dim=1))
    u, v, p = u_v_p[:, 0:1], u_v_p[:, 1:2], u_v_p[:, 2:3]  # Make sure to split correctly
    
    return u-0, v-0, p-0  # Return as separate tensors


def train_step(X_left, X_right, X_top, X_bottom, X_collocation, X_ic, optimizer, model, pde, max_grad_norm=1.0):
    """
    Perform a single training step for the Physics-Informed Neural Network (PINN).

    Parameters:
    X_left (torch.Tensor): Boundary points at the left side of the domain.
    X_right (torch.Tensor): Boundary points at the right side of the domain.
    X_top (torch.Tensor): Boundary points at the top side of the domain.
    X_bottom (torch.Tensor): Boundary points at the bottom side of the domain.
    X_collocation (torch.Tensor): Collocation points inside the domain for evaluating the PDE residuals.
    X_ic (torch.Tensor): Initial condition points.
    optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
    model (nn.Module): The PINN model which outputs the predicted values of u, v, and p.
    pde (function): Function to compute the residuals of the PDEs.
    max_grad_norm (float): Maximum gradient norm for gradient clipping. Default is 1.0.

    Returns:
    tuple: A tuple containing the losses:
        - BC_left_loss (torch.Tensor): Boundary condition loss at the left side.
        - BC_right_loss (torch.Tensor): Boundary condition loss at the right side.
        - BC_top_loss (torch.Tensor): Boundary condition loss at the top side.
        - BC_bottom_loss (torch.Tensor): Boundary condition loss at the bottom side.
        - pde_loss (torch.Tensor): PDE residual loss.
        - IC_loss (torch.Tensor): Initial condition loss.
        - loss (torch.Tensor): Total weighted loss.

    This function performs the following steps:
    1. Zero the gradients of the optimizer.
    2. Compute the initial condition loss using the initial_conditions function.
    3. Compute the boundary condition loss at the left, right, top, and bottom boundaries using the boundary_conditions function.
    4. Compute the PDE residual loss using the pde function.
    5. Compute the total weighted loss.
    6. Perform backpropagation and gradient clipping.
    7. Update the model parameters using the optimizer.

    Example:
        BC_left_loss, BC_right_loss, BC_top_loss, BC_bottom_loss, pde_loss, IC_loss, loss = train_step(X_left, X_right, X_top, X_bottom, X_collocation, X_ic, optimizer, model, pde)
    """
    optimizer.zero_grad()

    u_ic, v_ic, p_ic = initial_conditions(model, X_ic[:, 0:1], X_ic[:, 1:2])
    IC_loss = torch.mean(torch.square(u_ic)) + torch.mean(torch.square(v_ic)) + torch.mean(torch.square(p_ic))

    X_left[:, 0:1].requires_grad_(True)
    X_left[:, 1:2].requires_grad_(True)
    bc_inlet, _, bc_walls_left = boundary_conditions(model, X_left[:, 0:1], X_left[:, 1:2])
    BC_left_loss = torch.mean(torch.square(bc_inlet)) + torch.mean(torch.square(bc_walls_left))

    X_right[:, 0:1].requires_grad_(True)
    X_right[:, 1:2].requires_grad_(True)
    _, bc_outlet, bc_walls_right = boundary_conditions(model, X_right[:, 0:1], X_right[:, 1:2])
    BC_right_loss = torch.mean(torch.square(bc_outlet)) + torch.mean(torch.square(bc_walls_right))

    X_top[:, 0:1].requires_grad_(True)
    X_top[:, 1:2].requires_grad_(True)
    _, _, bc_walls_top = boundary_conditions(model, X_top[:, 0:1], X_top[:, 1:2])
    BC_top_loss = torch.mean(torch.square(bc_walls_top))

    X_bottom[:, 0:1].requires_grad_(True)
    X_bottom[:, 1:2].requires_grad_(True)
    _, _, bc_walls_bottom = boundary_conditions(model, X_bottom[:, 0:1], X_bottom[:, 1:2])
    BC_bottom_loss = torch.mean(torch.square(bc_walls_bottom))

    f_u, f_v, f_c = pde(X_collocation, model)
    pde_loss = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v)) + torch.mean(torch.square(f_c))

    loss = IC_loss + BC_left_loss + BC_right_loss + BC_top_loss + BC_bottom_loss +  pde_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    
    return BC_left_loss, BC_right_loss, BC_top_loss, BC_bottom_loss, pde_loss, IC_loss, loss


device = torch.device('cpu')
print(device)


# Parameters
N_space = 20
N_time = 20

# Spatial and temporal points
x_space = np.linspace(0, 1, N_space)
y_space = np.linspace(0, 1, N_space)
x_time = np.linspace(0, 1, N_time)

# Collocation points (interior points)
x_collocation, y_collocation, t_collocation = np.meshgrid(x_space[1:-1], y_space[1:-1], x_time[1:-1])
X_collocation = torch.tensor(np.hstack((x_collocation.reshape(-1, 1), y_collocation.reshape(-1, 1), t_collocation.reshape(-1, 1))), dtype=torch.float32, device=device)

# Initial condition points
x_ic, y_ic = np.meshgrid(x_space, y_space)
t_ic = np.zeros_like(x_ic)
X_ic = torch.tensor(np.hstack((x_ic.reshape(-1, 1), y_ic.reshape(-1, 1), t_ic.reshape(-1, 1))), dtype=torch.float32, device=device)

# Left boundary points
x_left = np.zeros_like(y_space)
y_left = np.linspace(0, 1, N_space)
t_left = np.zeros_like(y_left)
X_left = torch.tensor(np.hstack((x_left.reshape(-1, 1), y_left.reshape(-1, 1), t_left.reshape(-1, 1))), dtype=torch.float32, device=device)

# Right boundary points
x_right = np.full_like(y_space, 1)
y_right = np.linspace(0, 1, N_space)
t_right = np.zeros_like(x_right)
X_right = torch.tensor(np.hstack((x_right.reshape(-1, 1), y_right.reshape(-1, 1), t_right.reshape(-1, 1))), dtype=torch.float32, device=device)

# Top boundary points
x_walls_top = np.linspace(0, 1, N_space)
y_walls_top = np.full_like(x_walls_top, 1)
t_walls_top = np.zeros_like(x_walls_top)
X_top = torch.tensor(np.hstack((x_walls_top.reshape(-1, 1), y_walls_top.reshape(-1, 1), t_walls_top.reshape(-1, 1))), dtype=torch.float32, device=device)

# Bottom boundary points
x_walls_bottom = np.linspace(0, 1, N_space)
y_walls_bottom = np.zeros_like(x_walls_bottom)
t_walls_bottom = np.zeros_like(x_walls_bottom)
X_bottom = torch.tensor(np.hstack((x_walls_bottom.reshape(-1, 1), y_walls_bottom.reshape(-1, 1), t_walls_bottom.reshape(-1, 1))), dtype=torch.float32, device=device)

# Combine all boundary points
X_boundary = torch.cat((X_ic, X_left, X_right, X_top, X_bottom), dim=0)

# Randomly sample test points in [0, 1]^2
N_test = 100
x_test = np.random.rand(N_test)
y_test = np.random.rand(N_test)
t_test = np.zeros_like(x_test)
X_test = torch.tensor(np.hstack((x_test.reshape(-1, 1), y_test.reshape(-1, 1), t_test.reshape(-1, 1))), dtype=torch.float32, device=device)


# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_collocation.cpu().numpy()[:, 0], X_collocation.cpu().numpy()[:, 1], X_collocation.cpu().numpy()[:, 2], label='Collocation Points', color='blue')
ax.scatter(X_boundary.cpu().numpy()[:, 0], X_boundary.cpu().numpy()[:, 1], X_boundary.cpu().numpy()[:, 2], label='Boundary Points', color='red')
ax.scatter(X_test.cpu().numpy()[:, 0], X_test.cpu().numpy()[:, 1], X_test.cpu().numpy()[:, 2], label='Test Points', color='green')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.set_title('Collocation, Boundary, and Test Points')
plt.savefig(os.path.join(figures_dir, 'points.png'))

# Select the minimum size among all the tensors
min_size = min(X_left.size(0), X_right.size(0), X_top.size(0), X_bottom.size(0), X_collocation.size(0), X_ic.size(0))

# Trim the tensors to the minimum size
X_left = X_left[:min_size]
X_right = X_right[:min_size]
X_top = X_top[:min_size]
X_bottom = X_bottom[:min_size]
X_collocation = X_collocation[:min_size]
X_ic = X_ic[:min_size]

# Create a TensorDataset with the trimmed tensors
dataset = TensorDataset(X_left, X_right, X_top, X_bottom, X_collocation, X_ic)

# Create a DataLoader to load the data in batches
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the activation functions to use
activation_functions = {
    'SiLU': nn.SiLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'Leaky ReLU': nn.LeakyReLU()
}


# Placeholder for storing results
results = {}
models = {}

# Training loop for each activation function
for name, activation in activation_functions.items():
    print(f"Training with {name} activation function...")
    
    # Initialize the model and optimizer for each activation function.
    model = PINN(num_inputs=3, num_layers=5, num_neurons=60, activation=activation, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    # Time the training process
    start_time = time.time()

    epochs = 15000  # Number of training epochs
    loss_list = []  # List to store the average loss at each epoch
    epoch_list = []  # List to store the epoch number

    IC_weight = 1.0  # Weight for the initial condition loss
    BC_right_left = 1.0  # Weight for the boundary condition loss
    PDE_weight = 1.0  # Weight for the PDE loss

    print("Initial learning rate:", optimizer.param_groups[0]['lr'])  # Print the initial learning rate

    for epoch in range(epochs):
        total_loss = 0  # Variable to store the total loss for the epoch

        # Loop through the data loader to get the left, right, top, bottom, collocation, and initial condition points
        for batch in data_loader:
            X_left_batch, X_right_batch, X_top_batch, X_bottom_batch, X_collocation_batch, X_ic_batch = [b.to(device) for b in batch]

            # Perform a training step and get the individual losses and total loss for the epoch 
            BC_left_loss, BC_right_loss, BC_top_loss, BC_bottom_loss, pde_loss, IC_loss, loss_total = train_step(
                X_left_batch, X_right_batch, X_top_batch, X_bottom_batch, X_collocation_batch, X_ic_batch, optimizer, model, pde,
                max_grad_norm=1.0)
            
            total_loss += loss_total.item()  # Accumulate the total loss for the epoch

        average_loss = total_loss / len(data_loader)  # Calculate the average loss for the epoch
        loss_list.append(average_loss)  # Append the average loss to the loss list
        epoch_list.append(epoch)  # Append the epoch number to the epoch list

        if epoch % 100 == 0:
            scheduler.step()  # Adjust the learning rate using the scheduler
            print(f'Epoch {epoch}, Total loss {average_loss:.4e}')  #
            print(f'PDE loss: {pde_loss:.4e}, IC loss: {IC_loss:.4e}, BC left loss: {BC_left_loss:.4e}, BC right loss: {BC_right_loss:.4e}, BC top loss: {BC_top_loss:.4e}, BC bottom loss: {BC_bottom_loss:.4e}')  # Print the individual losses
            print("Learning rate:", optimizer.param_groups[0]['lr']) 

    end_time = time.time()  # Get the end time of the training process
    elapsed_time = end_time - start_time  
    minutes, seconds = divmod(elapsed_time, 60)
    print(f'Total time taken for {name} training: {int(minutes)} minutes {seconds:.2f} seconds')

    with open(f'runtime_{name}.txt', 'w') as file:
        file.write(f'Total time taken for training: {int(minutes)} minutes {seconds:.2f} seconds\n')
    
    # Store the results
    results[name] = {
        'loss_list': loss_list,
        'epoch_list': epoch_list
    }

    # Store the trained model
    models[name] = model


# Plotting the results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Mean Squared Error for Different Activation Functions')

axes = axes.flatten()
for ax, (name, data) in zip(axes, results.items()):
    ax.plot(data['epoch_list'], data['loss_list'], label=name)
    ax.set_title(name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(figures_dir, 'activation_functions_comparison.png'))
plt.close()


# Plotting the solution
N_space_plot = 100
N_time_plot = 100
x_space_plot = np.linspace(0, 1, N_space_plot)
y_space_plot = np.linspace(0, 1, N_space_plot)
x_time_plot = np.linspace(0, 1, N_time_plot)
x_space_mesh, y_space_mesh, x_time_mesh = np.meshgrid(x_space_plot, y_space_plot, x_time_plot)
x = np.hstack((x_space_mesh.reshape(-1, 1), y_space_mesh.reshape(-1, 1), x_time_mesh.reshape(-1, 1)))
x_tensor = torch.tensor(x, dtype=torch.float32, device=device)

time_steps = [0,  int(N_time_plot / 2), N_time_plot - 1]

for name, model in models.items():
    y_pred = model(x_tensor).detach().numpy().reshape(N_space_plot, N_space_plot, N_time_plot, 3)
    u_pred = y_pred[..., 0]
    v_pred = y_pred[..., 1]
    p_pred = y_pred[..., 2]

    for t in time_steps:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        c0 = axs[0].contourf(x_space_plot, y_space_plot, u_pred[:, :, t], levels=50)
        cb0 = fig.colorbar(c0, ax=axs[0])
        cb0.set_label('u velocity')
        axs[0].set_title(f'u velocity at t={x_time_plot[t]:.2f}')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')

        c1 = axs[1].contourf(x_space_plot, y_space_plot, v_pred[:, :, t], levels=50)
        cb1 = fig.colorbar(c1, ax=axs[1])
        cb1.set_label('v velocity')
        axs[1].set_title(f'v velocity at t={x_time_plot[t]:.2f}')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')

        c2 = axs[2].contourf(x_space_plot, y_space_plot, p_pred[:, :, t], levels=50)
        cb2 = fig.colorbar(c2, ax=axs[2])
        cb2.set_label('Pressure')
        axs[2].set_title(f'Pressure at t={x_time_plot[t]:.2f}')
        axs[2].set_xlabel('x')
        axs[2].set_ylabel('y')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{name}_solution_t_{x_time_plot[t]:.2f}.png'))
        plt.close()
