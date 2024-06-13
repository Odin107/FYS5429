import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
L = 2.0  # Length of the channel
H = 1.0  # Height of the channel
P1 = 8.0  # Pressure at the inlet
P2 = 0.0  # Pressure at the outlet
mu = 1.0  # Dynamic viscosity
Re = 100  # Reynolds number
rho = Re * mu  # Density for Re = 100

# Number of points in space
N_space = 100

# Spatial coordinates
y = np.linspace(0, H, N_space)
x = np.linspace(0, L, N_space)

# Pressure gradient
dp_dx = (P2 - P1) / L

# Analytical solution for u velocity (steady-state)
u_analytical = (-1 / (2 * mu) * dp_dx) * (y * (H - y))

# Create 2D arrays for the heatmap
u_analytical_2d = np.tile(u_analytical, (N_space, 1)).T
v_analytical_2d = np.zeros((N_space, N_space))
p_analytical_2d = np.tile(P1 + dp_dx * x, (N_space, 1))

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot u velocity heatmap
c0 = axs[0].contourf(x, y, u_analytical_2d, levels=50)
fig.colorbar(c0, ax=axs[0])
axs[0].set_title('u velocity (Analytical)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

# Plot v velocity heatmap (which is zero)
c1 = axs[1].contourf(x, y, v_analytical_2d, levels=50)
fig.colorbar(c1, ax=axs[1])
axs[1].set_title('v velocity (Analytical)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

# Plot pressure heatmap
c2 = axs[2].contourf(x, y, p_analytical_2d, levels=50)
fig.colorbar(c2, ax=axs[2])
axs[2].set_title('Pressure (Analytical)')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')

# Adjust layout
plt.tight_layout()

# Save the plots
figures_dir = "PINNs_NS/figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
plt.savefig(os.path.join(figures_dir, 'analytical_solution_heatmaps.png'))

# Show plots
plt.show()
