import cv2
import numpy as np
import os

def process_image(image_path, block_size=75, black_threshold=50, top_n=10):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray_image.shape
    
    # Create a copy of the original image to draw rectangles
    output_image = image.copy()
    
    # Initialize list to store block information
    block_info = []

    # Process image in blocks
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            # Extract block
            block = gray_image[y:y + block_size, x:x + block_size]
            color_block = image[y:y + block_size, x:x + block_size]
            
            # Calculate standard deviation of the block
            std_dev = np.std(block)
            
            # Calculate average RGB value of the block
            avg_rgb = np.mean(color_block, axis=(0, 1))
            avg_r, avg_g, avg_b = avg_rgb
            
            # Calculate distance from black
            distance_from_black = np.sqrt(avg_r**2 + avg_g**2 + avg_b**2)
            
            # Filter out blocks that are close to black
            if distance_from_black > black_threshold:
                # Store block information
                block_info.append((std_dev, (x, y, block_size, block_size)))
    
    # Sort blocks by standard deviation (ascending)
    block_info.sort(key=lambda x: x[0])
    
    # Select the top N blocks with the least standard deviation
    top_blocks = block_info[:top_n]
    
    # Draw rectangles around the selected blocks
    for _, (x, y, w, h) in top_blocks:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color
    
    # Display the result
    cv2.imshow('Processed Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for root, dirs, files in os.walk('images'):
    for file in files:
        # Construct the full file path
        file_path = os.path.join(root, file)
        
        process_image(file_path)
