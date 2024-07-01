import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import os

# Define sign language classes
sign_language_classes = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello', 27: 'How_are_you', 28: 'I_love_you',
    29: 'More', 30: 'Name', 31: 'Please', 32: 'Sorry', 33: 'Thank_you', 34: 'Yes', 35: 'No'
}

# Define directories for training and validation
train_dir = 'C:/Users/HP/OneDrive/Documents/python programs/internship/sign_language_dataset/train'
validation_dir = 'C:/Users/HP/OneDrive/Documents/python programs/internship/sign_language_dataset/validation'

# Create model function
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(len(sign_language_classes), activation='softmax')  # Output layer for ASL classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model function
def train_model():
    global model

    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        result_label.config(text="Error: Training or validation directory does not exist.")
        return

    # Image data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'
    )

    model = create_model()
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32
    )

    model.save('asl_model.h5')
    result_label.config(text="Model trained and saved as 'asl_model.h5'.")

# Process image function
def process_image(file_path):
    # Read the image using OpenCV
    image = cv2.imread(file_path)
    # Preprocess the image
    image = cv2.resize(image, (64, 64))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    # Get predicted sign language word
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_word = sign_language_classes.get(predicted_index, "Unknown")
    return predicted_word

# Upload file function
def upload_file():
    file_path = filedialog.askopenfilename()
    return file_path

# Process uploaded file function
def process_uploaded_file():
    global processed_image
    file_path = upload_file()
    if file_path:
        if file_path.endswith('.jpg') or file_path.endswith('.png'):
            processed_image = file_path
            result_label.config(text=f"Processing image: {file_path}")
        else:
            result_label.config(text="Unsupported file format. Please upload a supported image file.")

# Show result function
def show_result():
    if processed_image is not None:
        predicted_word = process_image(processed_image)
        result_label.config(text=f"Predicted Sign Language Word: {predicted_word}")
    else:
        result_label.config(text="No image available for prediction.")

# Load trained model function
def load_trained_model():
    global model
    try:
        model = load_model("asl_model.h5")
        result_label.config(text="Model loaded successfully.")
    except Exception as e:
        result_label.config(text="Model not found. Please train the model first.")
        model = None

# Initialize global variable for storing processed image
processed_image = None

# Create the Tkinter window
window = tk.Tk()
window.title("Sign Language Recognition")

# Set window size and background color
window.geometry("600x400")
window.configure(bg="#f0f0f0")

# Create a "Upload" button
upload_button = tk.Button(
    window,
    text="Upload File",
    command=process_uploaded_file,
    font=("Arial", 16, 'bold'),
    bg='#4CAF50',  # Green background color
    fg='white',    # White text color
    padx=20,       # Padding X
    pady=10,       # Padding Y
    relief='raised',  # Raised border for 3D effect
    bd=5            # Border width
)
upload_button.pack(pady=10)

# Create a "Show Result" button
result_button = tk.Button(
    window,
    text="Show Result",
    command=show_result,
    font=("Arial", 16, 'bold'),
    bg='#FFC107',  # Yellow background color
    fg='black',    # Black text color
    padx=20,       # Padding X
    pady=10,       # Padding Y
    relief='raised',  # Raised border for 3D effect
    bd=5            # Border width
)
result_button.pack(pady=10)

# Create a "Train Model" button
train_button = tk.Button(
    window,
    text="Train Model",
    command=train_model,
    font=("Arial", 16, 'bold'),
    bg='#009688',  # Teal background color
    fg='white',    # White text color
    padx=20,       # Padding X
    pady=10,       # Padding Y
    relief='raised',  # Raised border for 3D effect
    bd=5            # Border width
)
train_button.pack(pady=10)

# Create a "Load Model" button
load_model_button = tk.Button(
    window,
    text="Load Model",
    command=load_trained_model,
    font=("Arial", 16, 'bold'),
    bg='#FF5722',  # Orange background color
    fg='white',    # White text color
    padx=20,       # Padding X
    pady=10,       # Padding Y
    relief='raised',  # Raised border for 3D effect
    bd=5            # Border width
)
load_model_button.pack(pady=10)

# Create a label to display the result
result_label = tk.Label(
    window,
    text="",
    font=("Arial", 14, 'bold'),
    bg="#f0f0f0",
    fg="#333333"
)
result_label.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()
