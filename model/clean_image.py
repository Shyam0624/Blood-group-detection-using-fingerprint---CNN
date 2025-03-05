import cv2
import numpy as np
import os

# Load the JPEG image
image_path = "scan7b.jpg"  # Replace with your actual image path
image = cv2.imread(image_path)

# Step 1: Denoising the Image (Reduces JPEG artifacts)
denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Step 2: Sharpen the Image
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])  # Kernel for sharpening
sharpened_image = cv2.filter2D(denoised_image, -1, kernel_sharpening)

# Step 3: Adjust Brightness and Contrast
alpha = 1.2  # Contrast control (1.0-3.0)
beta = 20    # Brightness control (0-100)
adjusted_image = cv2.convertScaleAbs(sharpened_image, alpha=alpha, beta=beta)

# Step 4: Resize the Image to 128x128
resized_image = cv2.resize(adjusted_image, (128, 128), interpolation=cv2.INTER_AREA)

# Step 5: Save the Enhanced and Resized Image
folder_path = os.path.dirname(image_path)  # Get folder of the original image
output_path1 = os.path.join(folder_path, "scan7b_resized.bmp")  # Same folder, new name
output_path2 = os.path.join(folder_path, "scan7b_new.bmp")  # Same folder, new name
cv2.imwrite(output_path2, image)
#cv2.imwrite(output_path2, adjusted_image)

"""# Display Original, Enhanced, and Resized Image
cv2.imshow("Original Image", image)
cv2.imshow("Enhanced Image", adjusted_image)
cv2.imshow("Resized Image (128x128)", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

