import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Función de Rosenbrock
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Derivada parcial de la función de Rosenbrock respecto a x
def rosenbrock_dx(x, y):
    return -2 * (1 - x) - 400 * x * (y - x**2)

# Derivada parcial de la función de Rosenbrock respecto a y
def rosenbrock_dy(x, y):
    return 200 * (y - x**2)

# Función Esférica
def spherical(x, y):
    return x**2 + y**2

# Derivada parcial de la función esférica respecto a x
def spherical_dx(x, y):
    return 2 * x

# Derivada parcial de la función esférica respecto a y
def spherical_dy(x, y):
    return 2 * y

# Función de Ackley
def ackley(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20

# Derivada parcial de la función de Ackley respecto a x
def ackley_dx(x, y):
    return 2 * np.pi * np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) * np.sin(2*np.pi*x) - 0.2 * x / np.sqrt(0.5 * (x**2 + y**2))

# Derivada parcial de la función de Ackley respecto a y
def ackley_dy(x, y):
    return 2 * np.pi * np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) * np.sin(2*np.pi*y) - 0.2 * y / np.sqrt(0.5 * (x**2 + y**2))

# Función de Booth
def booth(x, y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

# Derivada parcial de la función de Booth respecto a x
def booth_dx(x, y):
    return 2 * (x + 2*y - 7) + 2 * (2*x + y - 5) * 2

# Derivada parcial de la función de Booth respecto a y
def booth_dy(x, y):
    return 2 * (x + 2*y - 7) * 2 + 2 * (2*x + y - 5)


# Algoritmo de gradiente descendente
def grad_desc(func, func_dx, func_dy, x_start, y_start, learning_rate, iterations):
    x = x_start
    y = y_start
    trajectory = [(x, y)]

    for _ in range(iterations):
        x -= learning_rate * func_dx(x, y)
        y -= learning_rate * func_dy(x, y)
        trajectory.append((x, y))

    return trajectory

# Configuración de los parámetros
learning_rate = 0.01
iterations = 1000
x_start = np.random.uniform(-10, 10)
y_start = np.random.uniform(-10, 10)

# Optimización de la función de Rosenbrock
trajectory = grad_desc(spherical, spherical_dx, spherical_dy, x_start, y_start, learning_rate, iterations)

# Configuración de la figura y el gráfico en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear la malla de puntos para la superficie
x_vals = np.linspace(-10, 10, 1000)
y_vals = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x_vals, y_vals)
Z = spherical(X, Y)

# Graficar la superficie
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Inicialización de la trayectoria de optimización
line, = ax.plot([], [], [], 'ro')

# Función de actualización de la animación
def update(frame):
    x, y = trajectory[frame]
    z = spherical(x, y)
    line.set_data([x], [y])
    line.set_3d_properties([z])
    return line,

# Crear la animación
ani = FuncAnimation(fig, update, frames=len(trajectory), blit=True)

# Mostrar la animación
plt.show()

