import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to train and evaluate the model
def train_and_evaluate(model, train_gen, valid_gen, epochs):
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    return history

# Plot training history for comparison
def plot_history(history, model_name, plot_type):
    plt.plot(history.history[plot_type], label=f'{model_name} {plot_type}')
    plt.plot(history.history[f'val_{plot_type}'], label=f'{model_name} Val {plot_type}')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epochs')
    plt.ylabel(plot_type.capitalize())
    plt.legend()
    plt.show()

# Evaluate and plot results for a model
def evaluate_and_plot(model, test_generator, model_name):
    print(f"Evaluating {model_name}...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"{model_name} Test Loss: {test_loss}")
    print(f"{model_name} Test Accuracy: {test_accuracy}")

    # Predict on test data
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    # Classification Report
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(f"{model_name} Classification Report:\n", report)

# Load your test dataset
test_data_dir = "C:/Users/shyam/Desktop/Final Mini Proj/data"  # Update path
img_height, img_width = 128, 128  # Ensure dimensions match the model
batch_size = 32

# Create test data generator
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load models
baseline_model = load_model('baseline_model.keras')
resnet_model = load_model('resnet_model.keras')
mobilenet_model = load_model('mobilenet_model.keras')

# Evaluate and plot results for each model
evaluate_and_plot(baseline_model, test_generator, "Baseline CNN")
evaluate_and_plot(resnet_model, test_generator, "ResNet50")
evaluate_and_plot(mobilenet_model, test_generator, "MobileNet")

# Plot training histories if available
if 'history_naive' in locals() and 'history_resnet' in locals() and 'history_mobilenet' in locals():
    # Plot accuracy
    plot_history(history_naive, 'Baseline CNN', plot_type='accuracy')
    plot_history(history_resnet, 'ResNet50', plot_type='accuracy')
    plot_history(history_mobilenet, 'MobileNet', plot_type='accuracy')

    # Plot loss
    plot_history(history_naive, 'Baseline CNN', plot_type='loss')
    plot_history(history_resnet, 'ResNet50', plot_type='loss')
    plot_history(history_mobilenet, 'MobileNet', plot_type='loss')
else:
    print("Training histories not available for plotting.")
