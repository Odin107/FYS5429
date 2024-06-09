import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
a = 5
D = 1 # Diffusion coefficient
t_max = 1.0  # Maximum time for the animation
frames = 100  # Number of frames in the animation
interval = 100  # Interval between frames in milliseconds

x = np.linspace(-5, 5, 500)  # Increased range and resolution
y = np.linspace(-5, 5, 500)  # Increased range and resolution
X, Y = np.meshgrid(x, y)

def initial_condition(x, y, a):
    return np.exp(-a * (x**2 + y**2))

def u(x, y, t, D):
    if t == 0:
        return initial_condition(x, y, a)
    else:
        return (1 / (4 * np.pi * D * t)) * np.exp(-(x**2 + y**2) / (4 * D * t))

def u_1d(x, t, D):
    if t == 0:
        return np.exp(-a * x**2)
    else:
        return (1 / np.sqrt(4 * np.pi * D * t)) * np.exp(-x**2 / (4 * D * t))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('2D Diffusion of a Gaussian Hill')

ax2.set_xlim([-5, 5])
ax2.set_ylim([0, 1])
ax2.set_xlabel('x')
ax2.set_ylabel('u(x, t)')
ax2.set_title('1D Diffusion along x-axis')

# Initial plot for 2D
Z = initial_condition(X, Y, a)
contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=0, vmax=1)
fig.colorbar(contour, ax=ax1)

# Initial plot for 1D
line, = ax2.plot(x, u_1d(x, 0, D))

def animate(frame):
    t = (frame + 1) * t_max / frames  # Ensure t starts at a non-zero value

    # Update 2D plot
    Z = u(X, Y, t, D)
    ax1.clear()
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis', vmin=0, vmax=1)
    ax1.set_title(f'2D Diffusion at t = {t:.2f}')
    ax1.set_xlim([-5, 5])
    ax1.set_ylim([-5, 5])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Update 1D plot
    ax2.clear()
    line, = ax2.plot(x, u_1d(x, t, D))
    ax2.set_title(f'1D Diffusion at t = {t:.2f}')
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x, t)')

    return contour, line

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval)

plt.show()
