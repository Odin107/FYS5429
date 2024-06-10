import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from torch.utils.data import DataLoader, TensorDataset

figures_dir = "PINNs_NS/figures"
os.makedirs(figures_dir, exist_ok=True)

class PINN(nn.Module):
    def __init__(self, num_inputs, num_layers, num_neurons, device):
        super(PINN, self).__init__()
        self.device = device
        activation = nn.SiLU()

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

def IC(x, model):
    u, v, p = torch.split(model(x), 1, dim=1)
    u_parabel = 4 * x[:, 1:2] * (1 - x[:, 1:2])
    v_zero = torch.zeros_like(v)
    p_zero = torch.full_like(p, 8.0)
    return u - u_parabel, v - v_zero, p - p_zero



def BC_left(x, model):
    u, v, p = torch.split(model(x), 1, dim=1)
    p_in = torch.zeros_like(p)
    v_zero = torch.zeros_like(v)
    return u , v - v_zero, p - p_in



def BC_right(x, model):
    u, v, p = torch.split(model(x), 1, dim=1)
    v_zero = torch.zeros_like(v)
    p_zero = torch.zeros_like(p)
    return u , v - v_zero, p - p_zero



def BC_top(x, model):
    u, v, p = torch.split(model(x), 1, dim=1)
    u_zero = torch.zeros_like(u)
    v_zero = torch.zeros_like(v)
    return u - u_zero, v - v_zero


def BC_bottom(x, model):
    u, v, p = torch.split(model(x), 1, dim=1)
    u_zero = torch.zeros_like(u)
    v_zero = torch.zeros_like(v)
    return u - u_zero, v - v_zero


def train_step(X_left, X_right, X_top, X_bottom, X_collocation, X_ic, optimizer, model, max_grad_norm=1.0, IC_weight=1.0, BC_weight=1.0, PDE_weight=1.0):
    optimizer.zero_grad()

    u_ic, v_ic, p_ic = IC(X_ic, model)
    IC_loss = torch.mean(torch.square(u_ic)) + torch.mean(torch.square(v_ic)) + torch.mean(torch.square(p_ic))

    u_left, v_left, p_left = BC_left(X_left, model)
    BC_left_loss = torch.mean(torch.square(u_left)) + torch.mean(torch.square(v_left)) + torch.mean(torch.square(p_left))

    u_right, v_right, p_right = BC_right(X_right, model)
    BC_right_loss = torch.mean(torch.square(u_right)) + torch.mean(torch.square(v_right)) + torch.mean(torch.square(p_right))

    u_top, v_top = BC_top(X_top, model)
    BC_top_loss = torch.mean(torch.square(u_top)) + torch.mean(torch.square(v_top))

    u_bottom, v_bottom = BC_bottom(X_bottom, model)
    BC_bottom_loss = torch.mean(torch.square(u_bottom)) + torch.mean(torch.square(v_bottom))

    f_u, f_v, f_c = pde(X_collocation, model)
    pde_loss = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v)) + torch.mean(torch.square(f_c))

    loss = IC_weight * IC_loss + BC_weight * (BC_left_loss + BC_right_loss + BC_top_loss + BC_bottom_loss) + PDE_weight * pde_loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return BC_left_loss, BC_right_loss, BC_top_loss, BC_bottom_loss, pde_loss, IC_loss, loss



device = torch.device('cpu')
print(device)


N_space = 20
N_time = 20


x_space = np.linspace(0, 1, N_space)
y_space = np.linspace(0, 1, N_space)
x_time = np.linspace(0, 1, N_time)
x_collocation, y_collocation, t_collocation = np.meshgrid(x_space[1:-1], y_space[1:-1], x_time[1:-1])
X_collocation = torch.tensor(np.hstack((x_collocation.reshape(-1, 1), y_collocation.reshape(-1, 1), t_collocation.reshape(-1, 1))), dtype=torch.float32, device=device)

x_ic, y_ic = np.meshgrid(x_space, y_space)
t_ic = np.zeros_like(x_ic)
X_ic = torch.tensor(np.hstack((x_ic.reshape(-1, 1), y_ic.reshape(-1, 1), t_ic.reshape(-1, 1))), dtype=torch.float32, device=device)

x_left = np.zeros_like(y_space)
y_left = np.linspace(0, 1, N_space)
t_left = np.zeros_like(y_left)
X_left = torch.tensor(np.hstack((x_left.reshape(-1, 1), y_left.reshape(-1, 1), t_left.reshape(-1, 1))), dtype=torch.float32, device=device)


x_right = np.full_like(y_space, 1)
y_right = np.linspace(0, 1, N_space)
t_right = np.zeros_like(x_right)
X_right = torch.tensor(np.hstack((x_right.reshape(-1, 1), y_right.reshape(-1, 1), t_right.reshape(-1, 1))), dtype=torch.float32, device=device)
x_walls_top = np.linspace(0, 1, N_space)
y_walls_top = np.full_like(x_walls_top, 1)
t_walls_top = np.zeros_like(x_walls_top)
X_top = torch.tensor(np.hstack((x_walls_top.reshape(-1, 1), y_walls_top.reshape(-1, 1), t_walls_top.reshape(-1, 1))), dtype=torch.float32, device=device)

x_walls_bottom = np.linspace(0, 1, N_space)
y_walls_bottom = np.zeros_like(x_walls_bottom)
t_walls_bottom = np.zeros_like(x_walls_bottom)
X_bottom = torch.tensor(np.hstack((x_walls_bottom.reshape(-1, 1), y_walls_bottom.reshape(-1, 1), t_walls_bottom.reshape(-1, 1))), dtype=torch.float32, device=device)

# Ensuring all tensors are of the same size
min_size = min(X_left.size(0), X_right.size(0), X_top.size(0), X_bottom.size(0), X_collocation.size(0), X_ic.size(0))

X_left = X_left[:min_size]
X_right = X_right[:min_size]
X_top = X_top[:min_size]
X_bottom = X_bottom[:min_size]
X_collocation = X_collocation[:min_size]
X_ic = X_ic[:min_size]

dataset = TensorDataset(X_left, X_right, X_top, X_bottom, X_collocation, X_ic)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PINN(num_inputs=3, num_layers=5, num_neurons=30, device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.96)


start_time = time.time()

epochs = 1000
loss_list = []
epoch_list = []

IC_weight = 2.0
BC_weight = 1.0
PDE_weight = 10.0

print("Initial learning rate:", optimizer.param_groups[0]['lr'])
for epoch in range(epochs):
    total_loss = 0
    for batch in data_loader:
        X_left_batch, X_right_batch, X_top_batch, X_bottom_batch, X_collocation_batch, X_ic_batch = [b.to(device) for b in batch]

        BC_left_loss, BC_right_loss, BC_top_loss, BC_bottom_loss, pde_loss, IC_loss, loss_total = train_step(
            X_left_batch, X_right_batch, X_top_batch, X_bottom_batch, X_collocation_batch, X_ic_batch, optimizer, model, 
            max_grad_norm=1.0, IC_weight=IC_weight, BC_weight=BC_weight, PDE_weight=PDE_weight)
        
        total_loss += loss_total.item()

    average_loss = total_loss / len(data_loader)
    loss_list.append(average_loss)
    epoch_list.append(epoch)
    scheduler.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Total loss {average_loss:.4e}')
        print(f'PDE loss: {pde_loss:.4e}, IC loss: {IC_loss:.4e}, BC left loss: {BC_left_loss:.4e}, BC right loss: {BC_right_loss:.4e}, BC top loss: {BC_top_loss:.4e}, BC bottom loss: {BC_bottom_loss:.4e}')
        print("Learning rate:", optimizer.param_groups[0]['lr'])


end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time/60:.2f} min")

plt.plot(epoch_list, loss_list)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(figures_dir, 'loss.png'))
plt.close()

N_space_plot = 100
N_time_plot = 100
x_space_plot = np.linspace(0, 1, N_space_plot)
y_space_plot = np.linspace(0, 1, N_space_plot)
x_time_plot = np.linspace(0, 1, N_time_plot)
x_space_mesh, y_space_mesh, x_time_mesh = np.meshgrid(x_space_plot, y_space_plot, x_time_plot)
x = np.hstack((x_space_mesh.reshape(-1, 1), y_space_mesh.reshape(-1, 1), x_time_mesh.reshape(-1, 1)))
x_tensor = torch.tensor(x, dtype=torch.float32, device=device)

y_pred = model(x_tensor).detach().numpy().reshape(N_space_plot, N_space_plot, N_time_plot, 3)

u_pred = y_pred[..., 0]
v_pred = y_pred[..., 1]
p_pred = y_pred[..., 2]

time_steps = [0, int(N_time_plot / 4), int(N_time_plot / 2), int(3 * N_time_plot / 4), N_time_plot - 1]

for t in time_steps:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plotting u velocity
    c0 = axs[0].contourf(x_space_plot, y_space_plot, u_pred[:, :, t], levels=50)
    cb0 = fig.colorbar(c0, ax=axs[0])
    cb0.set_label('u velocity')
    axs[0].set_title(f'u velocity at t={x_time_plot[t]:.2f}')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    # Plotting v velocity
    c1 = axs[1].contourf(x_space_plot, y_space_plot, v_pred[:, :, t], levels=50)
    cb1 = fig.colorbar(c1, ax=axs[1])
    cb1.set_label('v velocity')
    axs[1].set_title(f'v velocity at t={x_time_plot[t]:.2f}')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')

    # Plotting pressure
    c2 = axs[2].contourf(x_space_plot, y_space_plot, p_pred[:, :, t], levels=50)
    cb2 = fig.colorbar(c2, ax=axs[2])
    cb2.set_label('Pressure')
    axs[2].set_title(f'Pressure at t={x_time_plot[t]:.2f}')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'solution_t_{x_time_plot[t]:.2f}.png'))
    plt.close()
