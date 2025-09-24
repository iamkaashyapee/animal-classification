import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify, send_file
import tensorflow as tf
import io
import base64
import cv2

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)

# --- Model and Class Names ---
try:
    # Load the trained model
    MODEL_PATH = 'final_best_animal_classifier.keras'
    model = tf.keras.models.load_model(MODEL_PATH)
except (IOError, ImportError) as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

# IMPORTANT: You must replace these with the actual class names from your training data
# The order must be the same as the one used during training.
CLASS_NAMES = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
    'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion',
    'Panda', 'Tiger', 'Zebr'
]

IMG_HEIGHT = 160
IMG_WIDTH = 160

# --- Preprocessing ---
def preprocess_image(image):
    """
    Preprocesses the image for the model.
    - Converts to RGB
    - Resizes to (IMG_HEIGHT, IMG_WIDTH)
    - Converts to numpy array
    - Rescales pixel values to [0, 1]
    - Adds a batch dimension
    """
    img = image.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    # Get base model and last conv layer
    base_model = model.get_layer('mobilenetv2_1.00_160')
    last_conv_layer = base_model.get_layer('Conv_1')
    # Build a model that maps input to the last conv layer output
    conv_model = tf.keras.models.Model(base_model.input, last_conv_layer.output)
    # Build a model that maps conv features to prediction
    head_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = head_input
    for layer in model.layers[2:]:  # skip input and base model
        x = layer(x)
    head_model = tf.keras.models.Model(head_input, x)
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = head_model(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index)

def superimpose_heatmap_on_image(image, heatmap, alpha=0.4):
    # image: PIL Image, heatmap: np.array (H, W)
    img = np.array(image.resize((IMG_WIDTH, IMG_HEIGHT)))
    if img.shape[-1] == 4:
        img = img[..., :3]
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload, prediction, and returns the result.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            # Open the image
            image = Image.open(file.stream)

            # Preprocess the image
            processed_image = preprocess_image(image)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            
            # Check if the index is valid
            if predicted_class_index < len(CLASS_NAMES):
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = float(prediction[0][predicted_class_index])
                
                # Return result as JSON
                return jsonify({
                    'predicted_class': predicted_class_name,
                    'confidence': f'{confidence:.2%}'
                })
            else:
                return jsonify({'error': 'Prediction index out of range.'}), 500

        except Exception as e:
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500
            
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/heatmap', methods=['POST'])
def heatmap():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    try:
        image = Image.open(file.stream).convert('RGB')
        processed_image = preprocess_image(image)
        heatmap, pred_index = make_gradcam_heatmap(processed_image, model)
        superimposed_img = superimpose_heatmap_on_image(image, heatmap)
        # Encode as PNG in memory
        _, buffer = cv2.imencode('.png', superimposed_img)
        io_buf = io.BytesIO(buffer)
        io_buf.seek(0)
        return send_file(io_buf, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the app
    # Use 0.0.0.0 to make it accessible on your network
    app.run(host='0.0.0.0', port=5000, debug=True) 