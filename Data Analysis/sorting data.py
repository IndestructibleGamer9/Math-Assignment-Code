import matplotlib.pyplot as plt
import pandas as pd

# Define the ranges for x and y
x_ranges = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
y_ranges = [(70, 83), (83, 96), (96, 109), (109, 122), (122, 135)]

# Define the function to determine the category
def determine_category(value, ranges):
    for i, (low, high) in enumerate(ranges, 1):
        if low <= value < high:
            return i
    return None  # If value doesn't fit in any range

# Read the data from the CSV file
data = pd.read_csv(r'2024 Student Data.csv')

# Convert x and y values to integers
x = data['Shoulder_to_wrist_cm'].astype(int)
y = data['Waist_to_floor_cm'].astype(int)

# Process and categorize the data points
categorized_data = []
frequency = {'XS': 0, 'S': 0, 'M': 0, 'L': 0, 'XL': 0}
category_labels = {1: 'XS', 2: 'S', 3: 'M', 4: 'L', 5: 'XL'}

for xi, yi in zip(x, y):
    x_cat = determine_category(xi, x_ranges)
    y_cat = determine_category(yi, y_ranges)
    
    if x_cat is None or y_cat is None:
        # Skip points that don't fit into any category
        continue
    
    final_cat = max(x_cat, y_cat)
    final_cat_label = category_labels[final_cat]
    categorized_data.append((xi, yi, final_cat_label))
    if final_cat_label:
        frequency[final_cat_label] += 1

# Display the categorized data
for data_point in categorized_data:
    print(f"Point ({data_point[0]}, {data_point[1]}) is in category {data_point[2]}")

# Display the frequencies of each category
print("\nFrequencies of each category:")
for category, count in frequency.items():
    print(f"Category {category}: {count}")

# Define consistent colors for categories
category_colors = {'XS': '#e73535', 'S': '#e7e435', 'M': '#3be735', 'L': '#359de7', 'XL': '#b635e7'}

# Plot the categorized data
plt.figure(figsize=(12, 6))

# Scatter plot
plt.subplot(1, 2, 1)
for category, color in category_colors.items():
    x_cat = [data_point[0] for data_point in categorized_data if data_point[2] == category]
    y_cat = [data_point[1] for data_point in categorized_data if data_point[2] == category]
    plt.scatter(x_cat, y_cat, color=color, label=f'Category {category}')

# Plot the line y = 2.8x - 51.8426
x_line = range(40, 90)
y_line = [2.8 * xi - 51.8426 for xi in x_line]
plt.plot(x_line, y_line, color='black', linestyle='--', label='y = 2.8x - 51.8426')

plt.xlabel('Shoulder to Wrist (cm)')
plt.ylabel('Waist to Floor (cm)')
plt.title('Categorized Data Points')
plt.legend()
plt.grid(True)

# Bar chart
plt.subplot(1, 2, 2)
categories = list(frequency.keys())
counts = list(frequency.values())
colors = [category_colors[cat] for cat in categories]
plt.bar(categories, counts, color=colors)
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.title('Frequency of Each Category')

plt.tight_layout()
plt.show()
