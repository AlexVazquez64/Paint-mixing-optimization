import numpy as np
from scipy.optimize import minimize
import tkinter as tk
from tkinter import colorchooser

# Define the available paint colors and their corresponding RGB values
paint_colors = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'yellow': np.array([255, 255, 0]),
    'white': np.array([255, 255, 255]),
    'black': np.array([0, 0, 0]),
}

def hex_to_triplet(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))


def triplet_to_hex(rgb_color):
    return f'#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}'

# Define the cost function to be minimized

def cost_function(x, desired_color, selected_paint_colors):
    mixed_color = np.dot(x, np.array(list(selected_paint_colors.values())))
    return np.linalg.norm(desired_color - mixed_color)


# Define a small positive value
EPSILON = 1e-5

# Define the constraints for the optimization problem
constraints = (
    # sum of mixing proportions should be 1
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    # mixing proportions should be non-negative
    {'type': 'ineq', 'fun': lambda x: x},
)

def optimize_color_mix():
    global desired_color_label
    global paint_colors_listbox
    global result_label
    global final_color_label

    desired_color = desired_color_label.cget('bg')
    desired_color = np.array(hex_to_triplet(desired_color))

    paint_colors_selected_indices = paint_colors_listbox.curselection()
    if not paint_colors_selected_indices:
        result_label['text'] = "Please select at least one paint color."
        return

    selected_paint_colors = {color: value for i, (color, value) in enumerate(
        paint_colors.items()) if i in paint_colors_selected_indices}

    initial_guess = np.ones(len(selected_paint_colors)) / \
        len(selected_paint_colors)
    bounds = [(0, 1)] * len(selected_paint_colors)

    result = minimize(cost_function, initial_guess, args=(desired_color, selected_paint_colors),
                      constraints=constraints, bounds=bounds, options={'maxiter': 10000, 'disp': True})

    result_text = []
    for color, proportion in zip(selected_paint_colors.keys(), result.x):
        result_text.append(f'{color.capitalize()}: {proportion * 100:.2f}%')

    result_label['text'] = "\n".join(result_text)

    final_color = np.dot(result.x, np.array(
        list(selected_paint_colors.values())))
    final_color_label['bg'] = triplet_to_hex(tuple(final_color.astype(int)))
    final_color_label['text'] = f'Final mixed color: {final_color}'


def choose_desired_color():
    color = colorchooser.askcolor()[1]
    if color:
        desired_color_label['bg'] = color


root = tk.Tk()
root.title("Color Mixer")

# Desired color
tk.Label(root, text="Desired color:").grid(row=0, column=0, sticky='w')
desired_color_label = tk.Label(root, width=10, bg='white')
desired_color_label.grid(row=0, column=1, padx=5)
tk.Button(root, text="Choose color", command=choose_desired_color).grid(
    row=0, column=2, padx=5)

# Paint colors listbox
tk.Label(root, text="Available paint colors:").grid(
    row=1, column=0, sticky='w')
paint_colors_listbox = tk.Listbox(
    root, selectmode=tk.MULTIPLE, exportselection=False)
paint_colors_listbox.grid(row=1, column=1, padx=5)
for color in paint_colors.keys():
    paint_colors_listbox.insert(tk.END, color.capitalize())

# Optimize button
tk.Button(root, text="Optimize color mix", command=optimize_color_mix).grid(
    row=2, column=0, columnspan=3, pady=10)

# Results
result_label = tk.Label(root, text="")
result_label.grid(row=3, column=0, columnspan=3)

final_color_label = tk.Label(root, width=75)
final_color_label.grid(row=10, column=0, columnspan=3, pady=5)

root.mainloop()
