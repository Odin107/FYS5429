import matplotlib as mpl
import pyvista
import ufl
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 100
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])],[nx, ny], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Create initial condition
def initial_condition(x, a=5):
    """
    Create the initial condition for a heat equation problem.

    Parameters:
    x : array-like
        A 2D coordinate array, where x[0] represents the x-coordinate and x[1] represents the y-coordinate.
    a : float, optional
        A parameter controlling the spread of the initial condition. Default value is 5.

    Returns:
    float
        The value of the initial condition at the given coordinate.
    """
    return np.exp(-a * (x[0]**2 + x[1]**2))


u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)
# Create an XDMF file for storing the mesh and solution data
xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)

# Define the weak form of the heat equation
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

# Create the bilinear and linear forms
bilinear_form = fem.form(a)
linear_form = fem.form(L)

# Assemble the stiffness matrix and load vector
A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

# Create and configure the linear solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Start a virtual framebuffer for headless rendering
pyvista.start_xvfb()

# Create a PyVista grid for visualization
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

# Create a PyVista plotter and open a GIF file for saving the animation
plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif", fps=10)

# Add the grid to the plotter and configure the visualization settings
grid.point_data["uh"] = uh.x.array
warped = grid.warp_by_scalar("uh", factor=1)

viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                            cmap=viridis, scalar_bar_args=sargs,
                            clim=[0, max(uh.x.array)])

# Time-stepping loop
for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve the linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update the solution at the previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write the solution to the XDMF file
    xdmf.write_function(uh, t)

    # Update the plot
    new_warped = grid.warp_by_scalar("uh", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["uh"][:] = uh.x.array
    plotter.write_frame()

# Close the plotter and XDMF file
plotter.close()
xdmf.close()



