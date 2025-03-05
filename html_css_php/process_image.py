import numpy as np
import sys
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Mapping of blood groups to class indices
blood_group_set = {'A+': 0, 'A-': 1, 'AB+': 2, 'AB-': 3, 'B+': 4, 'B-': 5, 'O+': 6, 'O-': 7}
class_indices_to_blood_group = {v: k for k, v in blood_group_set.items()}  # Reverse dictionary to map index to blood group

# Set UTF-8 as the default encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load model
resnet_model = load_model('C:/Users/shyam/Desktop/Final Mini Proj/baseline_model.keras')

# Load and preprocess the image
image_path = sys.argv[1]
image = load_img(image_path, target_size=(128, 128))  # Update to match your model's input
image_array = img_to_array(image) / 255.0
image_array = image_array.reshape((1,) + image_array.shape)

# Make prediction
prediction = resnet_model.predict(image_array)
print(prediction)
predicted_class_index = np.argmax(prediction, axis=1)[0]
predicted_blood_group = class_indices_to_blood_group.get(predicted_class_index, "Unknown")

# Ensure predictions are printed safely
print(str(predicted_blood_group).encode('utf-8', errors='ignore').decode('utf-8'))
