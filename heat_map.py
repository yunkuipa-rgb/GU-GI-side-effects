import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the base medical image
base_image_path = "/Users/a14038/Documents/workspace/research_medical/dvh/MRI.png"  # Replace with your actual file path
base_image = Image.open(base_image_path).convert("L")  # Convert to grayscale if needed
base_image = np.array(base_image)

# Create an irregular heatmap with a small area of interest
rows, cols = base_image.shape
heatmap = np.zeros((rows, cols))

# Define the center and dimensions of the irregular area of interest
center_x, center_y = cols // 2 + 10, rows // 2 - 5  # Shift slightly right-down
radius_x, radius_y = 10, 20  # Width and height of the irregular region

# Generate an elliptical region with random noise for irregularity
x, y = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))
ellipse = ((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2) <= 1

# Add random noise to make the shape irregular
noise = np.random.rand(rows, cols) * 0.5
irregular_mask = ellipse & (noise > 0.3)  # Use threshold to create irregularity

# Assign high values to the region of interest
heatmap[irregular_mask] = 1.0

# Smooth the edges using a Gaussian-like filter
from scipy.ndimage import gaussian_filter
heatmap = gaussian_filter(heatmap, sigma=3)  # Adjust sigma for smoothness

# Normalize heatmap to range [0, 1]
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) - 0.5

# Apply a colormap to the heatmap
colormap = plt.cm.jet  # You can choose other colormaps like 'viridis' or 'hot'
heatmap_colored = colormap(heatmap)

# Overlay the heatmap on the base image
alpha = 0.7  # Transparency level of the heatmap
overlay_image = heatmap_colored[:, :, :3] * alpha + np.stack([base_image / 255.0] * 3, axis=-1) * (1 - alpha)

# Plot the result
plt.figure(figsize=(10, 10))
plt.imshow(overlay_image)
plt.axis("off")
plt.title("Irregular Heatmap Overlay")
plt.show()
