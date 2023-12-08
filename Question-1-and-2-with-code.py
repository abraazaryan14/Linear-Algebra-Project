#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to plot a plane based on a linear equation
def plot_plane(ax, coefficients, d, color):
    # Create a grid of x and y values
    x, y = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
    a, b, c = coefficients

    # Calculate z values based on the plane equation ax + by + cz = d
    z = (d - a * x - b * y) / c

    # Plot the plane
    ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, color=color, edgecolor='none')

# Function to create a 3D plot for a system of linear equations
def plot_system(equations):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each plane
    for eq in equations:
        plot_plane(ax, eq['coefficients'], eq['constant'], eq['color'])

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Plot of Linear Equations System')

    plt.show()

# Define your system of equations here
equations = [
    {'coefficients': [1, 1, 1], 'constant': 6, 'color': 'red'},
    {'coefficients': [-1, 1, 2], 'constant': 3, 'color': 'green'},
    {'coefficients': [2, -1, 1], 'constant': 1, 'color': 'gold'}
    # Add more equations as needed
]

# Plot the system
plot_system(equations)


# In[21]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to plot a plane based on a linear equation
def plot_plane(ax, coefficients, d, color):
    # Create a grid of x and y values
    x, y = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
    a, b, c = coefficients

    # Calculate z values based on the plane equation ax + by + cz = d
    z = (d - a * x - b * y) / c

    # Plot the plane
    ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, color=color, edgecolor='none')

# Function to create a 3D plot for a system of linear equations
def plot_system(equations):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each plane
    for eq in equations:
        plot_plane(ax, eq['coefficients'], eq['constant'], eq['color'])

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Plot of Linear Equations System')

    plt.show()

# Define your system of equations here
equations = [
    {'coefficients': [1, 1, 1], 'constant': 1, 'color': 'red'},
    {'coefficients': [1, 1, 1], 'constant': 7, 'color': 'green'},
    {'coefficients': [1, 1, 1], 'constant': 14, 'color': 'gold'}
    # Add more equations as needed
]

# Plot the system
plot_system(equations)


# In[13]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to plot a plane based on a linear equation
def plot_plane(ax, coefficients, d, color):
    # Create a grid of x and y values
    x, y = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
    a, b, c = coefficients

    # Calculate z values based on the plane equation ax + by + cz = d
    z = (d - a * x - b * y) / c

    # Plot the plane
    ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, color=color, edgecolor='none')

# Function to create a 3D plot for a system of linear equations
def plot_system(equations):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each plane
    for eq in equations:
        plot_plane(ax, eq['coefficients'], eq['constant'], eq['color'])

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Plot of Linear Equations System')

    plt.show()

# Define your system of equations here
equations = [
    {'coefficients': [1, -1, 1], 'constant': 4, 'color': 'red'},
    {'coefficients': [-1, 2, -1], 'constant': 1, 'color': 'green'},
    {'coefficients': [0, 1, 1], 'constant': 3, 'color': 'gold'}
    # Add more equations as needed
]

# Plot the system
plot_system(equations)


# In[17]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to plot a plane based on a linear equation
def plot_plane(ax, coefficients, d, color):
    # Create a grid of x and y values
    x, y = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
    a, b, c = coefficients

    # Calculate z values based on the plane equation ax + by + cz = d
    z = (d - a * x - b * y) / c

    # Plot the plane
    ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, color=color, edgecolor='none')

# Function to create a 3D plot for a system of linear equations
def plot_system(equations):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each plane
    for eq in equations:
        plot_plane(ax, eq['coefficients'], eq['constant'], eq['color'])

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Plot of Linear Equations System')

    plt.show()

# Define your system of equations here
equations = [
    {'coefficients': [1, 1, 1], 'constant': 5, 'color': 'red'},
    {'coefficients': [2, 2, 2], 'constant': 10, 'color': 'green'},
    {'coefficients': [4, 4, 4], 'constant': 20, 'color': 'gold'}
    # Add more equations as needed
]

# Plot the system
plot_system(equations)


# In[20]:


# Correct the code to adjust the position of the text label for y and include necessary imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Define the rotation matrix A(theta)
def A(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])

# Given vector x and angle theta
x = np.array([[1], [1]])
theta = 3 * np.pi / 4

# Apply the rotation matrix A to vector x
y = A(theta) @ x

# Plot the original and transformed vectors
plt.figure(figsize=(8, 8))
plt.grid(True)
plt.axis('equal')

# Origin point
origin = [0], [0]

# Plot the original vector x
plt.quiver(*origin, x[0,0], x[1,0], color='r', scale=1, scale_units='xy', angles='xy')
plt.text(x[0,0], x[1,0], 'x', color='r', fontsize=14)

# Plot the transformed vector y, with the label slightly above the endpoint
plt.quiver(*origin, y[0,0], y[1,0], color='b', scale=1, scale_units='xy', angles='xy')
plt.text(y[0,0], y[1,0] + 0.1, 'y', color='b', fontsize=14)  # Adjusted label position

# Arc for angle theta
arc = Arc((0, 0), 0.5, 0.5, angle=0, theta1=45, theta2=180, capstyle='round', linestyle='-', lw=2)
plt.gca().add_patch(arc)
plt.text(-0.25, 0.25, r'$\theta$', fontsize=14)

# Set the x and y axis limits
plt.xlim(-2, 2)
plt.ylim(-2, 2)

# Labels
plt.xlabel('X axis')
plt.ylabel('Y axis')

# Show the plot with the vectors and the rotation angle
plt.show()

# Output the rotated vector y
y


# In[ ]:




