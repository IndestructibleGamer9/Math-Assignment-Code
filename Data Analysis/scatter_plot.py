import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

# Start timer for operation
s = time.perf_counter()

# Load and clean data
data = pd.read_csv(r"C:\Users\Will\Documents\School Work\Math\Data Analysis\2024 Student Data.csv")
data = data.dropna()

# Convert to numeric and handle any errors
data['Shoulder_to_wrist_cm'] = pd.to_numeric(data['Shoulder_to_wrist_cm'], errors='coerce')
data['Waist_to_floor_cm'] = pd.to_numeric(data['Waist_to_floor_cm'], errors='coerce')

# Print statistical summary and correlation matrix
print(data.describe())
print(data.corr())

# Drop any rows with NaN values after conversion
data = data.dropna()

# Pair Plot using Matplotlib
def pair_plot(data):
    columns = data.columns
    num_columns = len(columns)
    fig, axes = plt.subplots(num_columns, num_columns, figsize=(15, 15))

    for i in range(num_columns):
        for j in range(num_columns):
            if i == j:
                axes[i, j].hist(data[columns[i]], bins=20)
                axes[i, j].set_title(columns[i])
            else:
                axes[i, j].scatter(data[columns[j]], data[columns[i]], alpha=0.5)
            if i == num_columns - 1:
                axes[i, j].set_xlabel(columns[j])
            if j == 0:
                axes[i, j].set_ylabel(columns[i])
    plt.tight_layout()
    plt.show()

pair_plot(data)

# Box Plots using Matplotlib
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.boxplot(data['Shoulder_to_wrist_cm'], vert=False)
plt.title('Box Plot of Shoulder to Wrist')
plt.subplot(122)
plt.boxplot(data['Waist_to_floor_cm'], vert=False)
plt.title('Box Plot of Waist to Floor')
plt.tight_layout()
plt.show()

# Prepare data for linear regression
X = data['Shoulder_to_wrist_cm'].values.reshape(-1, 1)
y = data['Waist_to_floor_cm'].values

# Train the model
start_time = time.time()
model = LinearRegression()
model.fit(X, y)
end_time = time.time()

# Print results
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Training time: {end_time - start_time} seconds")

# Create a smooth line for prediction
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)

# Plot the data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="black", alpha=0.5)
plt.plot(X_line, y_line, color="red")
plt.xlabel('Shoulder to Wrist (cm)')
plt.ylabel('Waist to Floor (cm)')
plt.title('Shoulder to Wrist vs Waist to Floor Measurements')
plt.tight_layout()
plt.show()

# Operation time
f = time.perf_counter()
print(f'Operation took {f-s} seconds')

# Histograms
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.hist(data['Shoulder_to_wrist_cm'], bins=20)
plt.title('Histogram of Shoulder to Wrist')
plt.subplot(122)
plt.hist(data['Waist_to_floor_cm'], bins=20)
plt.title('Histogram of Waist to Floor')
plt.tight_layout()
plt.show()

# Residual plot
residuals = y - model.predict(X)
plt.scatter(model.predict(X), residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

# Q-Q plot for residuals
fig, ax = plt.subplots(figsize=(6, 4))
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Q-Q plot")
plt.show()

# Polynomial regression
degrees = [1, 2, 3]
plt.figure(figsize=(12, 4))
for i, degree in enumerate(degrees):
    ax = plt.subplot(1, 3, i+1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.scatter(X, y)
    plt.plot(X_test, model.predict(X_test), color='r')
    plt.title(f'Polynomial Degree {degree}')
plt.tight_layout()
plt.show()

# Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X, y)
importances = rf.feature_importances_
feature_names = ['Shoulder_to_wrist_cm']

# Feature importances plot
plt.bar(feature_names, importances)
plt.title('Feature Importances')
plt.show()

# Heat Map using Matplotlib
corr = data.corr()
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(corr, cmap='coolwarm')
fig.colorbar(cax)
ticks = np.arange(0, len(corr.columns), 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)
plt.title('Heat Map of Correlations', pad=20)
plt.show()
