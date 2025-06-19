import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define ReLU function
def relu(x):
    return np.maximum(0, x)

# Setup plot
fig, ax = plt.subplots()
x_vals = np.linspace(-10, 10, 400)
y_vals = relu(x_vals)
ax.plot(x_vals, y_vals, color='green')
dot, = ax.plot([], [], 'ro')
text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

ax.set_xlim(-10, 10)
ax.set_ylim(-1, 11)
ax.set_xlabel("Input")
ax.set_ylabel("ReLU Output")
ax.set_title("ReLU Activation Function - Animated Simulation")
ax.grid(True)

# Initialize
def init():
    dot.set_data([], [])
    text.set_text('')
    return dot, text

# Animate
def animate(i):
    x = -10 + i * 0.2
    y = relu(x)
    dot.set_data([x], [y])
    text.set_text(f'Input: {x:.2f}, Output: {y:.2f}')
    return dot, text

# Create and save animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=100, blit=True)
ani.save("relu_activation_simulation.gif", writer='pillow')
plt.show()