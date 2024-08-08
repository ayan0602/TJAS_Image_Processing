import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('meningioma-sample1.png')


# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#trying to clean up image
blurred_img = cv2.medianBlur(gray_image, 5)

# Define the Sobel operator for vertical edges
sobel_vertical = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

# Define the Sobel operator for horizontal edges
sobel_horizontal = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

convoluted_image = blurred_img

# Apply the Sobel operator
vertical_edges = cv2.filter2D(convoluted_image, -1, sobel_vertical)

horizontal_edges = cv2.filter2D(convoluted_image, -1, sobel_horizontal)


# apply median blur to remove artifacts from edges
vertical_edges = cv2.medianBlur(vertical_edges, 15)

horizontal_edges = cv2.medianBlur(horizontal_edges, 15)

# Combine vertical and horizontal edges
combined_edges = np.sqrt(np.square(vertical_edges) + np.square(horizontal_edges))
combined_edges = np.uint8(combined_edges)

# Display the results
plt.figure(figsize=(18, 6))

# Display the original grayscale image
plt.subplot(1, 4, 1)
plt.title('Original Grayscale Image')
plt.imshow(convoluted_image, cmap='gray')
plt.axis('off')

# Display the vertical edges
plt.subplot(1, 4, 2)
plt.title('Vertical Edges')
plt.imshow(vertical_edges, cmap='gray')
plt.axis('off')

# Display the horizontal edges
plt.subplot(1, 4, 3)
plt.title('Horizontal Edges')
plt.imshow(horizontal_edges, cmap='gray')
plt.axis('off')


# Display the combined edges
plt.subplot(1, 4, 4)
plt.title('Combined Edges')
plt.imshow(combined_edges, cmap='gray')
plt.axis('off')

plt.show()
