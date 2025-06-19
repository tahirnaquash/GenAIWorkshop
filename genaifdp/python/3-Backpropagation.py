import numpy as np

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    return x * (1 - x)

# Input dataset (4 samples, 2 features)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
# Output dataset
y = np.array([[0], [1], [1], [0]])

# Initialize weights
np.random.seed(42)
input_layer_neurons = 2
hidden_layer_neurons = 3
output_neurons = 1

# Random weights and biases
w1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
b1 = np.zeros((1, hidden_layer_neurons))
w2 = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))

# Training loop
for epoch in range(10000):
    # Forward pass
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    # Backpropagation
    error = y - a2
    d2 = error * sigmoid_deriv(a2)
    d1 = d2.dot(w2.T) * sigmoid_deriv(a1)

    # Update weights and biases
    w2 += a1.T.dot(d2) * 0.1
    b2 += np.sum(d2, axis=0, keepdims=True) * 0.1
    w1 += X.T.dot(d1) * 0.1
    b1 += np.sum(d1, axis=0, keepdims=True) * 0.1

# Output after training
print("Final output after training:")
print(a2)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Use your trained weights and biases
# (Assume w1, b1, w2, b2 are already defined from your training code)

# Define sigmoid again for animation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Network structure
layer_sizes = [2, 3, 1]
positions = {
    0: [(-1, 1), (-1, -1)],  # Input layer
    1: [(0, 1.5), (0, 0), (0, -1.5)],  # Hidden layer
    2: [(1.5, 0)]  # Output layer
}

# Setup plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')

# Draw neurons
neurons = {}
for layer, coords in positions.items():
    for i, (x, y) in enumerate(coords):
        circle = plt.Circle((x, y), 0.2, color='lightgray', ec='black')
        ax.add_patch(circle)
        neurons[(layer, i)] = circle

# Draw connections
lines = []
for i, (x1, y1) in enumerate(positions[0]):
    for j, (x2, y2) in enumerate(positions[1]):
        line, = ax.plot([x1, x2], [y1, y2], 'gray', lw=1)
        lines.append(line)
for i, (x1, y1) in enumerate(positions[1]):
    for j, (x2, y2) in enumerate(positions[2]):
        line, = ax.plot([x1, x2], [y1, y2], 'gray', lw=1)
        lines.append(line)

# Input samples
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# Animation function
def animate(i):
    x = X[i % 4].reshape(1, -1)
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    # Update input layer colors
    for j in range(2):
        neurons[(0, j)].set_facecolor('skyblue' if x[0, j] > 0 else 'lightgray')

    # Update hidden layer colors
    for j in range(3):
        neurons[(1, j)].set_facecolor((1, 1 - a1[0, j], 1 - a1[0, j]))

    # Update output layer color
    neurons[(2, 0)].set_facecolor((1 - a2[0, 0], 1, 1 - a2[0, 0]))

    return list(neurons.values()) + lines

ani = FuncAnimation(fig, animate, frames=4, interval=1000, blit=True)
plt.show()