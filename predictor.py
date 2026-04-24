import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simulated Data: Atomic Mass vs Sound Velocity (Example)
# X: Atomic Mass (approx values for illustrative ML model)
X = np.array([[138.9], [140.1], [140.9], [151.9], [173.0]]) 
# y: Computed Sound Velocities (m/s) from MSMA
y = np.array([1462.48, 618.5, 923.29, 1205.61, 1164.47])

# Model Training
model = LinearRegression()
model.fit(X, y)

# Prediction for a hypothetical element
hypothetical_mass = np.array([[160.0]])
predicted_velocity = model.predict(hypothetical_mass)

print(f"Predicted Sound Velocity for Mass 160: {predicted_velocity[0]:.2f} m/s")

# Plotting the trend
plt.scatter(X, y, color='blue', label='MSMA Computed Data')
plt.plot(X, model.predict(X), color='red', linestyle='--', label='ML Linear Fit')
plt.title('ML Regression: Atomic Mass vs Sound Velocity')
plt.xlabel('Atomic Mass')
plt.ylabel('Sound Velocity (m/s)')
plt.legend()
plt.show()
