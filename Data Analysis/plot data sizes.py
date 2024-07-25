import matplotlib.pyplot as plt

# Data
sizes = {'XS': 2.14, 'S': 19.56, 'M': 28.36, 'L': 27.16, 'XL': 16.16}

# Colors
category_colors = {'XS': '#e73535', 'S': '#e7e435', 'M': '#3be735', 'L': '#359de7', 'XL': '#b635e7'}

# Labels and values
labels = list(sizes.keys())
values = list(sizes.values())
colors = [category_colors[label] for label in labels]

# Plot
plt.figure(figsize=(10, 7))
plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the plot
plt.title('Category Distribution')
plt.show()
