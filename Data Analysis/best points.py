import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

# Data
x = [57,54,60,54,62,56,55,50,53,57,52,51,61,63,52,61,52,58,53,57,56,56,50,62,54,53,56,46,75,59,73,74,59,59,49,63,56,64,48,58,64,61,65,53,69,49,56,49,54,56]
y = [107,100,116,100,103,106,107,97,108,111,95,110,108,119,96,106,99,104,102,108,100,105,94,106,111,107,90,84,95,92,102,98,107,98,88,106,109,101,78,110,84,105,123,95,111,97,111,105,84,91]

# Convert to numpy arrays
x = np.array(x)
y = np.array(y)

# Define the linear function
def linear_func(p, x):
    return p[0] * x + p[1]

# Create a model for fitting
linear_model = odr.Model(linear_func)

# Create a RealData object using our initiated data
data = odr.RealData(x, y)

# Set up ODR with the model and data
odr_obj = odr.ODR(data, linear_model, beta0=[1., 1.])

# Run the regression
out = odr_obj.run()

# Extract the parameters
slope = out.beta[0]
intercept = out.beta[1]

# Print results
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

# Create points for the line
x_line = np.array([min(x), max(x)])
y_line = slope * x_line + intercept

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5)
plt.plot(x_line, y_line, color='red', linewidth=2)

plt.xlabel('Shoulder to Wrist (cm)')
plt.ylabel('Waist to Floor (cm)')
plt.title('Shoulder to Wrist vs Waist to Floor Measurements (Deming Regression)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Print two points on the line
print(f"\nTwo points on the line:")
print(f"Point 1: ({x_line[0]:.2f}, {y_line[0]:.2f})")
print(f"Point 2: ({x_line[1]:.2f}, {y_line[1]:.2f})")

# Print all data points
print("\nAll data points:")
for i in range(len(x)):
    print(f"({x[i]}, {y[i]})")
    #print(f'{i} : {y[i]-x[i]}')