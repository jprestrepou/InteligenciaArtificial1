# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:22:45 2024

@author: jupa_
"""

import numpy as np
import matplotlib.pyplot as plt
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
learning_rate = 0.001
iterations = 1000
x_start = np.random.uniform(-2, 2)
y_start = np.random.uniform(-2, 2)

# Optimización de la función de Rosenbrock
trajectory = grad_desc(rosenbrock, rosenbrock_dx, rosenbrock_dy, x_start, y_start, learning_rate, iterations)

# Visualización del proceso de optimización
fig, ax = plt.subplots()
x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = rosenbrock(X, Y)

def update(frame):
    ax.clear()
    ax.contour(X, Y, Z, levels=50)
    ax.plot(*trajectory[frame], 'ro')
    ax.set_title(f"Iteration {frame}/{iterations}")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

ani = FuncAnimation(fig, update, frames=range(iterations), repeat=False)
plt.show()
