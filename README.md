import numpy as np
import matplotlib.pyplot as plt

# Define grid size and parameters
nx, ny = 50, 50  # Grid points in x and y directions
dx, dy = 1.0, 1.0  # Grid spacing

# Define charge distribution (source term ρ)
rho = np.zeros((nx, ny))
rho[nx//2, ny//2] = -1.0  # Point charge at the center

# Initialize potential field ϕ with zero
phi = np.zeros((nx, ny))

# Set boundary conditions (Dirichlet: phi = 0 at the edges)
phi[:, 0] = 0  # Left boundary
phi[:, -1] = 0  # Right boundary
phi[0, :] = 0  # Bottom boundary
phi[-1, :] = 0  # Top boundary

# Iterative solution using Gauss-Seidel method
tolerance = 1e-5  # Convergence criteria
max_iter = 5000  # Maximum iterations

for it in range(max_iter):
    old_phi = phi.copy()

    # Update interior points using finite difference
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - dx**2 * rho[i, j])

    # Check for convergence
    if np.max(np.abs(phi - old_phi)) < tolerance:
        print(f"Converged after {it} iterations.")
        break

# Plot the potential distribution
plt.figure(figsize=(8, 6))
plt.imshow(phi, extent=[0, nx, 0, ny], origin='lower', cmap='inferno')
plt.colorbar(label="Potential ϕ(x, y)")
plt.title("Solution of Poisson's Equation using Finite Difference")
plt.xlabel("x")
plt.ylabel("y")
plt.show()



