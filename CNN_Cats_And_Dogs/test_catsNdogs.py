# *********************** Test the model **************************
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("cats_vs_dogs_model.h5")

# Define a prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Resize
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)  # Predict
    print("Dog" if prediction[0] > 0.5 else "Cat")  # Threshold

# Test with a sample image
for i in range(0, 9):
    predict_image(f"dataset/test/cats/{i}.jpg")
    print(f"Image number: {i}")
    
# Test with a sample image
for i in range(0, 9):
    predict_image(f"dataset/test/dogs/{i}.jpg")
    print(f"Image number: {i}")