import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Example dataset: [size (sqft), number of rooms]
X = np.array([
    [2104, 3],
    [1600, 3],
    [2400, 3],
    [1416, 2],
    [3000, 4]
], dtype=float)

# House prices
y = np.array([399900, 329900, 369000, 232000, 539900], dtype=float)

# Normalize features
X = X / np.max(X, axis=0)
y = y / np.max(y)

# Build the model
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(2,)),
    layers.Dense(5, activation='relu'),
    layers.Dense(1)
])
   
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=1000, verbose=0)

# Test prediction
test = np.array([[2200, 2]], dtype=float)
test = test / np.max(X, axis=0)
predicted = model.predict(test)
print(f"Predicted price for [2200 sqft, 2 rooms]: {predicted[0][0] * np.max(y):.2f}")