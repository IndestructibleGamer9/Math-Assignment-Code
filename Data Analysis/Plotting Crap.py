import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'sleeve_length': (57,54,60,54,62,56,55,50,53,57,52,51,61,63,52,61,52,58,53,57,56,56,50,62,54,53,56,46,75,59,73,74,59,59,49,63,56,64,48,58,64,61,65,53,69,49,56,49,54,56),
    'trouser_length': (107,100,116,100,103,106,107,97,108,111,95,110,108,119,96,106,99,104,102,108,100,105,94,106,111,107,90,84,95,92,102,98,107,98,88,106,109,101,78,110,84,105,123,95,111,97,111,105,84,91)
})

# Set up the plot
plt.figure(figsize=(10, 10))
plt.style.use('ggplot')

# Create the scatter plot
plt.scatter(data['sleeve_length'], data['trouser_length'], color='black', s=20)


# Customize the plot
plt.title('Trouser length vs. sleeve length', fontsize=14)
plt.xlabel('Sleeve length (cm)', fontsize=12)
plt.ylabel('Trouser length (cm)', fontsize=12)

# Set axis limits and ticks
plt.xlim(40, 80)
plt.ylim(70, 150)
plt.xticks(np.arange(40, 81, 10))
plt.yticks(np.arange(70, 151, 13))

# Add colored background sections
colors = ['#FFFFE0', '#E0FFFF', '#E0FFE0', '#FFE0FF', '#FFE0E0']
for i, color in enumerate(colors):
    plt.axvspan(40+i*8, 40+(i+1)*8, facecolor=color, alpha=0.3)

# Add grid
plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

# Add text annotation (replace with your desired text and position)
plt.text(57, 105, '(57, 105)', ha='center', va='center', fontsize=10)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()