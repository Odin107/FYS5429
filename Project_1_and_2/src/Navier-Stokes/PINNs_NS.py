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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(42)

figures_dir = "PINNs_NS/figures"
os.makedirs(figures_dir, exist_ok=True)

class PINN(nn.Module):
    def __init__(self, num_inputs, num_layers, num_neurons, activation, device):
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
        return self.model(x)


def pde(x, model):
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
H = 1.0  # Height of the channel

def boundary_conditions(model, x, y):
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
    t = torch.zeros_like(x)
    u_v_p = model(torch.cat([x, y, t], dim=1))
    u, v, p = u_v_p[:, 0:1], u_v_p[:, 1:2], u_v_p[:, 2:3]  # Make sure to split correctly
    
    return u-0, v-0, p-0  # Return as separate tensors

def train_step(X_left, X_right, X_top, X_bottom, X_collocation, X_ic, optimizer, model, pde, max_grad_norm=1.0, IC_weight=4.0, BC_weight=10.0, PDE_weight=1.0):
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

    loss = IC_weight * IC_loss + BC_weight * (BC_left_loss + BC_right_loss) + BC_top_loss + BC_bottom_loss + PDE_weight * pde_loss

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

min_size = min(X_left.size(0), X_right.size(0), X_top.size(0), X_bottom.size(0), X_collocation.size(0), X_ic.size(0))

X_left = X_left[:min_size]
X_right = X_right[:min_size]
X_top = X_top[:min_size]
X_bottom = X_bottom[:min_size]
X_collocation = X_collocation[:min_size]
X_ic = X_ic[:min_size]

dataset = TensorDataset(X_left, X_right, X_top, X_bottom, X_collocation, X_ic)
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
    
    model = PINN(num_inputs=3, num_layers=8, num_neurons=60, activation=activation, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    start_time = time.time()

    epochs = 30000
    loss_list = []
    epoch_list = []

    IC_weight = 1.0
    BC_right_left = 1.0
    PDE_weight = 1.0

    print("Initial learning rate:", optimizer.param_groups[0]['lr'])

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            X_left_batch, X_right_batch, X_top_batch, X_bottom_batch, X_collocation_batch, X_ic_batch = [b.to(device) for b in batch]

            BC_left_loss, BC_right_loss, BC_top_loss, BC_bottom_loss, pde_loss, IC_loss, loss_total = train_step(
                X_left_batch, X_right_batch, X_top_batch, X_bottom_batch, X_collocation_batch, X_ic_batch, optimizer, model, pde,
                max_grad_norm=1.0, IC_weight=IC_weight, BC_weight=BC_right_left, PDE_weight=PDE_weight)
            
            total_loss += loss_total.item()

        average_loss = total_loss / len(data_loader)
        loss_list.append(average_loss)
        epoch_list.append(epoch)

        if epoch % 100 == 0:
            scheduler.step()
            print(f'Epoch {epoch}, Total loss {average_loss:.4e}')
            print(f'PDE loss: {pde_loss:.4e}, IC loss: {IC_loss:.4e}, BC left loss: {BC_left_loss:.4e}, BC right loss: {BC_right_loss:.4e}, BC top loss: {BC_top_loss:.4e}, BC bottom loss: {BC_bottom_loss:.4e}')
            print("Learning rate:", optimizer.param_groups[0]['lr'])

    end_time = time.time()
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

# Plotting solutions
N_space_plot = 100
N_time_plot = 100
x_space_plot = np.linspace(0, 1, N_space_plot)
y_space_plot = np.linspace(0, 1, N_space_plot)
x_time_plot = np.linspace(0, 1, N_time_plot)
x_space_mesh, y_space_mesh, x_time_mesh = np.meshgrid(x_space_plot, y_space_plot, x_time_plot)
x = np.hstack((x_space_mesh.reshape(-1, 1), y_space_mesh.reshape(-1, 1), x_time_mesh.reshape(-1, 1)))
x_tensor = torch.tensor(x, dtype=torch.float32, device=device)

time_steps = [0, int(N_time_plot / 4), int(N_time_plot / 2), int(3 * N_time_plot / 4), N_time_plot - 1]

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
