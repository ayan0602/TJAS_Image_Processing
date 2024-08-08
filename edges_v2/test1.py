import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


#plt.figure(figsize=(18, 12))

def clean_image(imagePath):
    image = cv2.imread(imagePath)

    # Specify the new size (e.g., half the original size)
    new_width = image.shape[1] // 5
    new_height = image.shape[0] // 5

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # Get the dimensions of the image
    height, width, channels = resized_image.shape

    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            # Get the BGR values (OpenCV uses BGR instead of RGB)
            b, g, r = resized_image[y, x]

            grey_scaled_average = ( r + g + b ) // 3

            print(f'avg: {grey_scaled_average}')

            resized_image[y, x] = [grey_scaled_average, grey_scaled_average, grey_scaled_average] 
            print(f' image y,x:  {resized_image[y, x]}')
            
            if grey_scaled_average < 50:
                resized_image[y, x] = [0, 0, 255]

            # Process or print the RGB values
            print(f'Pixel at ({x}, {y}): R={r}, G={g}, B={b}')
    
    cv2.imshow('colored pic',resized_image)
    cv2.waitKey(0)

def eff_clean_image(imagePath):
    # Load the image
    image = cv2.imread(imagePath)

    # Convert to grayscale using weighted average
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    # Calculate the average brightness of the grayscale image
    average_brightness = np.mean(gray_image)
    print(average_brightness)

    # Define the base threshold value
    base_threshold = 150

    # Adjust the threshold based on the average brightness
    if average_brightness < 90:  # If the image is darker
        # Lower the threshold value
        adjusted_threshold = 120
    else:  # If the image is lighter
        # Raise the threshold value
        adjusted_threshold = base_threshold

    # Create a mask where the grayscale values are above the threshold
    mask = gray_image > adjusted_threshold

    # Create an output image that starts as a copy of the original image
    output_image = image.copy()

    # Apply the mask to the output image: set pixels above the threshold to red
    output_image[mask] = [0, 0, 255]  # BGR value for red in OpenCV

    cv2.imshow('colored pic', output_image)
    cv2.waitKey(0)

def contour_image(imagePath):
    # Load the image
    image = cv2.imread(imagePath)
    if image is None:
            print(f"Error: Unable to load image at {imagePath}")
            return
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to create a binary image
    _, binary_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an output image
    output_image = image.copy()

    # Draw contours on the output image
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        print(area)
        
        # Define a minimum and maximum area for contours (you might need to adjust these)
        min_area = 1000  # Minimum area to filter out noise
        max_area = 90000 # Maximum area to filter out large structures like the skull

        # Calculate contour bounding box
        x, y, w, h = cv2.boundingRect(contour)
        location = (x, y, w, h)
        
        print(w*h)

        print(location)

        if min_area < area < max_area:
            # Draw the contour in red on the output image
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)  # Red color in BGR


    # Display the original and processed images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Binary Image')
    plt.imshow(binary_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Processed Image with Contours')
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def render_edges(imagePath, sobel_vertical, sobel_horizontal, medianBlurAmount, kernel, rows, cols) -> None:
    # get image
    image = cv2.imread(imagePath)
    # make image into greyscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # could do some pre-processing here to clean up image 
    # all the effects will be stored in 'processed_image'
    # not yet implemented 

    # no preprocessing yet so the only processing the image gets is the greyscale effect
    processed_image = cv2.blur(gray_image, kernel)

    # more robust edge detection algorithm
    canny_edges = cv2.Canny(gray_image, 50, 150)

    # ------------------------------------------- old edge detection algorithm  -------------------------------------------------------------

    # get the vertical and horizontal edges of the image
    vert_edges = cv2.filter2D(processed_image, -1, sobel_vertical)
    hori_edges = cv2.filter2D(processed_image, -1, sobel_horizontal)

    # remove artifacts from edge detection
    processed_vert_edges = cv2.medianBlur(vert_edges, medianBlurAmount)
    processed_hori_edges = cv2.medianBlur(hori_edges, medianBlurAmount)

    # combine the horizontal and vertical edges into one image
    combined_edges = np.sqrt(np.square(processed_vert_edges) + np.square(processed_hori_edges))
    combined_edges = np.uint8(combined_edges)
    # ---------------------------------------------------------------------------------------------------------------------

    # plot results 
    #plt.figure(figsize=plotSize)


    img_name = imagePath.split('.')
    img_name = img_name[0]

    # Get the dimensions of the image
    height, width, channels = image.shape

    # Display the original image before any processing
    plt.subplot(rows, cols, 1)
    plt.title(f' {img_name}')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # Display the image after processing
    plt.subplot(rows, cols, 2)
    plt.title('Processed Image')
    plt.imshow(processed_image, cmap='gray')
    plt.axis('off')

     # Display the vertical edges
    plt.subplot(rows, cols, 3)
    plt.title('Vertical Edges')
    plt.imshow(vert_edges, cmap='gray')
    plt.axis('off')

    # Display the horizontal edges
    plt.subplot(rows, cols, 4)
    plt.title('Horizontal Edges')
    plt.imshow(hori_edges, cmap='gray')
    plt.axis('off')
    
    # Display the canny edges
    plt.subplot(rows, cols, 5)
    plt.title('combined Edges')
    plt.imshow(combined_edges, cmap='gray')
    plt.axis('off')


    # Display the canny edges
    plt.subplot(rows, cols, 6)
    plt.title('Canny Edges')
    plt.imshow(canny_edges, cmap='gray')
    plt.axis('off')


'''
render_edges('images/m-s4.png', 
            np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]),
            np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]]),
            15,
            (15,15),
            rows=1, cols=6)

render_edges('images/meningioma-sample2.png', 
            np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]),
            np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]]),
            15,
            (20,20),
            rows=2, cols=6)

plt.subplots_adjust(hspace=1) 
plt.show()

'''


'''
# Iterate through each file in the directory
for root, dirs, files in os.walk('images'):
    for file in files:
        # Construct the full file path
        file_path = os.path.join(root, file)
        
        eff_clean_image(file_path)

'''

for root, dirs, files in os.walk('images'):
    for file in files:
        # Construct the full file path
        file_path = os.path.join(root, file)
        
        contour_image(file_path)
