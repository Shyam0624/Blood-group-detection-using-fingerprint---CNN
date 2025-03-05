from tensorflow.keras.models import load_model
import random
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import numpy as np

# Load models
baseline_model = load_model('baseline_model.keras')
resnet_model = load_model('resnet_model.keras')
mobilenet_model = load_model('mobilenet_model.keras')


# Mapping of blood groups to class indices
blood_group_set = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}
class_indices_to_blood_group = {v: k for k, v in blood_group_set.items()}  # Reverse dictionary to map index to blood group

# Path to the testing directory
dir_path = r"C:\Users\shyam\Desktop\Final Mini Proj\data"

# Get a list of all subdirectories (classes)
subfolders = [os.path.join(dir_path, subfolder) for subfolder in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subfolder))]

# Select a random subfolder
random_subfolder = random.choice(subfolders)
blood_group_from_folder = os.path.basename(random_subfolder) 

# Get a list of all image files in the selected subfolder
image_files = [os.path.join(random_subfolder, f) for f in os.listdir(random_subfolder) if f.endswith(('.png', '.jpg', '.jpeg','.BMP'))]

# Select a random image
random_image_path = random.choice(image_files)

# Load and display the image
img = image.load_img(random_image_path, target_size=(128, 128))
plt.imshow(img)
plt.axis('off')
plt.title(f"Random Image from Blood Group Folder: {blood_group_from_folder}")
plt.show()

# Preprocess the image
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)
X = X / 255.0  # Rescale to match the training data preprocessing

# Predict the class of the image
val = baseline_model.predict(X)

# Get the predicted class index
predicted_class_index = np.argmax(val, axis=1)[0]

# Map the predicted index to the corresponding blood group
predicted_blood_group = class_indices_to_blood_group.get(predicted_class_index, "Unknown")

# Get the corresponding probability value for the predicted class
predicted_probability = val[0][predicted_class_index]

# Print the result
print(f"\nThe image was randomly chosen from the '{blood_group_from_folder}' folder.\n")
print(val)
print(f"\nThe model predicts this blood group to be: {predicted_blood_group} with a probability of {predicted_probability:.4f}")


