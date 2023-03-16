import numpy as np
from scipy.optimize import minimize

# Define the desired color in RGB format
desired_color = np.array([200, 100, 50])

# Define the available paint colors and their corresponding RGB values
paint_colors = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'yellow': np.array([255, 255, 0]),
    'purple': np.array([128, 0, 128]),
    'orange': np.array([255, 165, 0])
}

# Define the cost function to be minimized
def cost_function(x, desired_color, paint_colors):
    mixed_color = np.dot(x, paint_colors.values())
    return np.linalg.norm(desired_color - mixed_color)

# Define the constraints for the optimization problem
constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, # sum of mixing proportions should be 1
    {'type': 'ineq', 'fun': lambda x: x} # mixing proportions should be non-negative
)

# Define the initial guess for the mixing proportions
initial_guess = np.ones(len(paint_colors)) / len(paint_colors)

# Perform the optimization
result = minimize(cost_function, initial_guess, args=(desired_color, paint_colors), constraints=constraints)

# Print the mixing proportions for each paint color
for color, proportion in zip(paint_colors.keys(), result.x):
    print(f'{color.capitalize()}: {proportion * 100:.2f}%')

# Print the final mixed color
final_color = np.dot(result.x, paint_colors.values())
print('Final mixed color:', final_color)
