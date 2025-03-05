# General libraries
import os
import glob
import shutil
import warnings
import time
import itertools
import pathlib
import math

# Data handling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# Image handling and visualization
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


# Deep learning libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

# Define Constants
BATCH_SIZE = 32
IMG_SIZE_MOBILENET = (224, 224)
IMG_SIZE_RESNET = (224, 224)
IMG_SIZE_SEQUENTIAL=(128, 128)
EPOCHS = 30
NUM_CLASSES = 8

# Dataset directory path
file_path = r"C:\Users\shyam\Desktop\Final Mini Proj\data"

# List all .BMP files recursively
filepaths = glob.glob(os.path.join(file_path, '**', '*.BMP'), recursive=True)
labels = os.listdir(file_path)

# Extract labels (folder names) for each file path
labels_path = [os.path.basename(os.path.dirname(path)) for path in filepaths]

# Create a DataFrame with file paths and labels
df = pd.DataFrame({'filePath': filepaths, 'label': labels_path})

# Shuffle the DataFrame
df = df.sample(frac=1).reset_index(drop=True)
df.head()

# Display class distribution
def get_class_distribution(df):
    labels_count = df['label'].value_counts()
    total_samples = len(df)
    for label, count in labels_count.items():
        percentage = round(count / total_samples * 100, 2)
        print(f"{label:<20s}:   {count} or {percentage}%")

get_class_distribution(df)
# Visualize distribution
def plot_class_distribution(df, label="label"):
    plt.figure(figsize=(12, 4))
    blood_type_counts = df[label].value_counts()
    colors = plt.cm.tab10(range(len(blood_type_counts)))
    plt.bar(blood_type_counts.index, blood_type_counts.values, color=colors)
    plt.xlabel('Blood Type')
    plt.ylabel('Frequency')
    plt.title('Distribution of Blood Types')
    plt.show()

plot_class_distribution(df)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['label'])

# Check encoded labels
df.head()

# Function to sample images and labels
def sample_images_data(data, labels, num_samples=1):
    sample_images = []
    sample_labels = []

    # Iterate through each label and sample a few images
    # Changed: Iterate directly through the labels list
    for label in labels:
        samples = data[data['label'] == label].head(num_samples)
        for j in range(len(samples)):
            img_path = samples.iloc[j]['filePath']
            img = Image.open(img_path).convert("RGB")  # Open the image and convert to RGB
            img = img.resize((128, 128))  # Resize image to target size
            img_array = np.array(img)  # Convert image to numpy array
            sample_images.append(img_array)
            sample_labels.append(samples.iloc[j]['label'])

    print(f"Total number of sample images to plot: {len(sample_images)}")
    return sample_images, sample_labels

# Plot sample images
def plot_images(images, labels, cmap="Blues"):
    f, ax = plt.subplots(2, 4, figsize=(8, 4))
    for i, img in enumerate(images):
        ax[i//4, i%4].imshow(img, cmap=cmap)
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title(labels[i])
    plt.show()

# Sample images from dataset and plot them
sample_images, sample_labels = sample_images_data(df, labels)
plot_images(sample_images, sample_labels)

# Apply oversampling to balance the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df[['filePath']], df['category_encoded'])

# Create a new DataFrame with oversampled data
df_resampled = pd.DataFrame(X_resampled, columns=['filePath'])
df_resampled['category_encoded'] = y_resampled

# Check new class distribution
print("\nClass distribution after oversampling:")
print(df_resampled['category_encoded'].value_counts())

# Visualize new class distribution
plot_class_distribution(df_resampled, "category_encoded")

# Split data into training, validation, and test sets
train_df, temp_df = train_test_split(
    df_resampled,
    train_size=0.8,
    stratify=df_resampled['category_encoded'],
    random_state=42
    )

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['category_encoded'],
    random_state=42
    )

# Check the shape of the splits
train_df.shape, valid_df.shape, test_df.shape


#For Sequential 
sequential_train_datagen = ImageDataGenerator(rescale=1./255)
sequential_valid_datagen = ImageDataGenerator(rescale=1./255)
sequential_test_datagen = ImageDataGenerator(rescale=1./255)

# Convert 'category_encoded' to string explicitly
train_df['category_encoded'] = train_df['category_encoded'].astype(str)
valid_df['category_encoded'] = valid_df['category_encoded'].astype(str)
test_df['category_encoded'] = test_df['category_encoded'].astype(str)
# Check if the conversion was successful
print(train_df['category_encoded'].dtype)  # Should output: object (string type)


sequential_train_gen = sequential_train_datagen.flow_from_dataframe(
    train_df, x_col='filePath', y_col='category_encoded',
    target_size=IMG_SIZE_SEQUENTIAL, batch_size=BATCH_SIZE, class_mode='categorical')

sequential_valid_gen = sequential_valid_datagen.flow_from_dataframe(
    valid_df, x_col='filePath', y_col='category_encoded',
    target_size=IMG_SIZE_SEQUENTIAL, batch_size=BATCH_SIZE, class_mode='categorical')

sequential_test_gen = sequential_test_datagen.flow_from_dataframe(
    test_df, x_col='filePath', y_col='category_encoded',
    target_size=IMG_SIZE_SEQUENTIAL, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

#For mobilenet
mobilenet_train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)
mobilenet_valid_datagen = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)
mobilenet_test_datagen = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)

mobilenet_train_gen = mobilenet_train_datagen.flow_from_dataframe(
    train_df, x_col='filePath', y_col='category_encoded',
    target_size=IMG_SIZE_MOBILENET, batch_size=BATCH_SIZE, class_mode='categorical', color_mode='rgb')

mobilenet_valid_gen = mobilenet_valid_datagen.flow_from_dataframe(
    valid_df, x_col='filePath', y_col='category_encoded',
    target_size=IMG_SIZE_MOBILENET, batch_size=BATCH_SIZE, class_mode='categorical')

mobilenet_test_gen = mobilenet_test_datagen.flow_from_dataframe(
    test_df, x_col='filePath', y_col='category_encoded',
    target_size=IMG_SIZE_MOBILENET, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)


#For resenet
resnet_train_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)
resnet_valid_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)
resnet_test_datagen = ImageDataGenerator(preprocessing_function=resnet_preprocess)

resnet_train_gen = resnet_train_datagen.flow_from_dataframe(
    train_df, x_col='filePath', y_col='category_encoded',
    target_size=IMG_SIZE_RESNET, batch_size=BATCH_SIZE, class_mode='categorical')

resnet_valid_gen = resnet_valid_datagen.flow_from_dataframe(
    valid_df, x_col='filePath', y_col='category_encoded',
    target_size=IMG_SIZE_RESNET, batch_size=BATCH_SIZE, class_mode='categorical')

resnet_test_gen = resnet_test_datagen.flow_from_dataframe(
    test_df, x_col='filePath', y_col='category_encoded',
    target_size=IMG_SIZE_RESNET, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)


def create_baseline_cnn(input_shape=(128, 128, 3), num_classes=8, learning_rate=0.001):
    """Baseline CNN Model"""
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_resnet_model(input_shape=(224, 224, 3), num_classes=8, learning_rate=1e-3):
    """ResNet50-based Model"""
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False  # Freeze base model 
    model = models.Sequential([
        base_model,
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=8, learning_rate=0.001):
    """MobileNet-based Model"""
    base_model = MobileNet(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
        
    model = models.Sequential([
        base_model,
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)


# Function to train and evaluate the model
def train_and_evaluate(model, train_gen, valid_gen, epochs):
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    return history


# Create models
baseline_model = create_baseline_cnn(input_shape=(128, 128, 3), num_classes=8)
resnet_model = create_resnet_model(input_shape=(224, 224, 3), num_classes=8)
mobilenet_model = create_mobilenet_model(input_shape=(224, 224, 3), num_classes=8)


# Train and evaluate models
print("Training Baseline CNN Model:")
history_naive = train_and_evaluate(baseline_model, sequential_train_gen, sequential_valid_gen, epochs=EPOCHS)

print("Training ResNet Model:")
history_resnet = train_and_evaluate(resnet_model, resnet_train_gen, resnet_valid_gen, epochs=EPOCHS)

print("Training MobileNet Model:")
history_mobilenet = train_and_evaluate(mobilenet_model, mobilenet_train_gen, mobilenet_valid_gen, epochs=EPOCHS)

# Plotting Training History for Comparison
def plot_history(history, model_name, plot_type):
    plt.plot(history.history[plot_type], label=f'{model_name} {plot_type}')
    plt.plot(history.history[f'val_{plot_type}'], label=f'{model_name} Val {plot_type}')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epochs')
    plt.ylabel(f'{plot_type}')
    plt.legend()
    plt.show()

# Plot accuracy for each model
plot_history(history_naive, 'Baseline CNN', plot_type='accuracy')
plot_history(history_resnet, 'ResNet50', plot_type='accuracy')
plot_history(history_mobilenet, 'MobileNet', plot_type='accuracy')

# Plot accuracy for each model
plot_history(history_naive, 'Baseline CNN', plot_type='loss')
plot_history(history_resnet, 'ResNet50', plot_type='loss')
plot_history(history_mobilenet, 'MobileNet', plot_type='loss')


baseline_model.summary()
resnet_model.summary()
mobilenet_model.summary()

baseline_model.save('baseline_model.keras')
resnet_model.save('resnet_model.keras')
mobilenet_model.save('mobilenet_model.keras') 