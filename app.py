import os
import logging
import numpy as np
import tensorflow as tf
import tifffile as tiff
from flask import Flask, render_template, request
import matplotlib.pyplot as plt

app = Flask(__name__)

tf.keras.config.enable_unsafe_deserialization()
model = tf.keras.models.load_model('best_model.keras')  # Load your trained model here

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def scale_images(images):
    return images / 255.0  # Scale image pixels to [0, 1]


def convert_to_binary(image, threshold=0.5):
    binary_image = (image > threshold).astype(np.uint8) * 255  # Convert to binary (0 and 255)
    return binary_image


def predict_and_process(model, images):
    images_tensor = tf.convert_to_tensor(images)
    outputs = model(images_tensor)

    probabilities = tf.nn.softmax(outputs, axis=-1)

    # Debugging logs
    print("Model outputs shape:", outputs.shape)
    print("Unique output values:", np.unique(outputs.numpy()))

    processed_preds = (probabilities > 0.5).numpy() * 255  # Binarize predictions and scale to 255

    # Debugging logs for processed predictions
    print("Unique processed predictions:", np.unique(processed_preds))

    return processed_preds


def process_image(filepath):
    image = tiff.imread(filepath)
    if image.ndim != 3 or image.shape[2] != 12:  # Ensure the image has 12 channels
        raise ValueError("Image must have 12 channels.")

    scaled_image = scale_images(image[np.newaxis, ...])  # Add batch dimension
    predicted_mask = predict_and_process(model, scaled_image)
    return predicted_mask[0]  # Return the predicted mask


logging.basicConfig(level=logging.DEBUG)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(uploaded_filepath)
            logging.debug(f"Uploaded image saved at: {uploaded_filepath}")

            try:
                image = tiff.imread(uploaded_filepath)
                logging.debug(f"Successfully opened image: {uploaded_filepath}")
            except Exception as e:
                logging.error(f"Error opening image: {e}")
                return "Error: Unable to open the uploaded image. Please upload a valid TIFF file.", 400

            # Convert the first channel of the uploaded image to binary (0s and 1s)
            binary_uploaded_image = convert_to_binary(image[..., 0])  # Taking the first channel

            # Save the binary uploaded image
            binary_uploaded_filename = os.path.splitext(file.filename)[0] + '_binary.png'
            binary_uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], binary_uploaded_filename)
            plt.imsave(binary_uploaded_filepath, binary_uploaded_image, cmap='gray')
            logging.debug(f"Converted binary uploaded image saved at: {binary_uploaded_filepath}")

            try:
                predicted_mask = process_image(uploaded_filepath)
            except ValueError as ve:
                logging.error(f"Processing error: {ve}")
                return str(ve), 400

            pred_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted.png')
            plt.imsave(pred_filepath, predicted_mask[..., 0], cmap='gray')  # Black-and-white predicted mask
            logging.debug(f"Predicted mask saved at: {pred_filepath}")

            return render_template('result.html', uploaded_image=binary_uploaded_filename,
                                   predicted_image='predicted.png')

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
