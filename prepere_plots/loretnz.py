import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz_system(t, y, sigma, rho, beta):
    """
    Define the Lorenz system of ordinary differential equations.

    Parameters:
    - t: Time variable (not used explicitly as the Lorenz system is autonomous)
    - y: Current state [x, y, z]
    - sigma, rho, beta: Parameters of the Lorenz system

    Returns:
    - dydt: Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = y
    dxdt = sigma * (y - x)
    dydt = rho * x - y - x * z
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_lorenz_trajectory(sigma, rho, beta, initial_conditions, t_span, num_points=10000):
    """
    Generate a trajectory of the Lorenz system using solve_ivp.

    Parameters:
    - sigma, rho, beta: Parameters of the Lorenz system
    - initial_conditions: Initial state [x0, y0, z0]
    - t_span: Time span of integration [t_start, t_end]
    - num_points: Number of points in the trajectory

    Returns:
    - sol: Solution object containing the trajectory
    """
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    sol = solve_ivp(lorenz_system, t_span, initial_conditions, args=(sigma, rho, beta), t_eval=t_eval, dense_output=True)
    return sol

def plot_lorenz_trajectory(sol):
    """
    Plot the trajectory of the Lorenz system.

    Parameters:
    - sol: Solution object containing the trajectory
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.y[2], label='Lorenz System')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Lorenz System Trajectory')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Example usage:
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    initial_conditions = [1.0, 0.0, 0.0]
    t_span = (0, 100)
    num_points = 100000

    sol = generate_lorenz_trajectory(sigma, rho, beta, initial_conditions, t_span, num_points)
    print(sol.y[2])
    plot_lorenz_trajectory(sol)
