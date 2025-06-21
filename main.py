import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the dental X-ray image
image = cv2.imread('xray.jpg', 0)

# Step 1: Preprocessing - Gaussian Blur
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Step 2: Histogram Equalization
equalized = cv2.equalizeHist(blurred)

# Step 3: Adaptive Thresholding
thresh = cv2.adaptiveThreshold(equalized, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

# Step 4: Morphological operations
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Step 5: Find contours
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 6: Draw contours on the original image
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, contours, -1, (0, 0, 255), 1)

# Create output folder if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save intermediate and final outputs
cv2.imwrite('output/01_original.png', image)
cv2.imwrite('output/02_blurred.png', blurred)
cv2.imwrite('output/03_equalized.png', equalized)
cv2.imwrite('output/04_thresholded.png', thresh)
cv2.imwrite('output/05_output_with_contours.png', output)

# Optional: display results using matplotlib
titles = ['Original', 'Blurred', 'Equalized', 'Thresholded', 'Final Output']
images = [image, blurred, equalized, thresh, output]

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray' if i != 4 else None)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.savefig('output/06_all_steps.png')
plt.show()
