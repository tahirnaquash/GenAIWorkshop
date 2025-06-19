import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Setup
y_true = 1
frames = 100
y_preds = np.linspace(-2, 3, 500)

# Loss functions
def compute_losses(y_pred):
    mse = (y_pred - y_true) ** 2
    mae = np.abs(y_pred - y_true)
    huber = np.where(np.abs(y_pred - y_true) < 1,
                     0.5 * (y_pred - y_true)**2,
                     np.abs(y_pred - y_true) - 0.5)
    return mse, mae, huber

# Initialize figure
fig, ax = plt.subplots(figsize=(10,6))
line1, = ax.plot([], [], 'b-', label='MSE (L2 Loss)')
line2, = ax.plot([], [], 'g--', label='MAE (L1 Loss)')
line3, = ax.plot([], [], 'r-.', label='Huber Loss')

ax.set_xlim(-2, 3)
ax.set_ylim(0, 4)
ax.set_title("Loss Functions Animation")
ax.set_xlabel("Prediction")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)

# Animation function
def animate(i):
    factor = i / frames
    dynamic_pred = y_preds * factor + y_true * (1 - factor)
    mse, mae, huber = compute_losses(dynamic_pred)
    line1.set_data(y_preds, mse)
    line2.set_data(y_preds, mae)
    line3.set_data(y_preds, huber)
    return line1, line2, line3

ani = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
plt.show()
