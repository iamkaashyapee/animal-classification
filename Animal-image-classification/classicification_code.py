# Animal Image Classification with Grad-CAM Integration

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

# 1. Path Setup

data_dir = r"C:\Users\ishuv\Downloads\Projects-20240722T093004Z-001\Projects\animal_classification\Animal Classification\dataset"

if not os.path.exists(data_dir):
    print(f"Error: Dataset directory not found at {data_dir}")
    exit()


# 2. Data Preparation

IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

if train_generator.num_classes == 0:
    print(f"Error: No classes found in dataset at: {data_dir}")
    exit()

num_classes = train_generator.num_classes
class_names = list(train_generator.class_indices.keys())

# 3. Model Architecture

base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Initial Training

initial_callbacks = [
    ModelCheckpoint("initial_best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
]

history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=initial_callbacks
)


# 5. Fine-Tuning

print("\n--- Starting Fine-tuning ---")
base_model.trainable = True
optimizer_fine = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer_fine, loss='categorical_crossentropy', metrics=['accuracy'])

final_callbacks = [
    ModelCheckpoint("final_best_animal_classifier.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
]

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    initial_epoch=history_initial.epoch[-1] + 1,
    callbacks=final_callbacks
)


# 6. Evaluation

print("\nEvaluating the final model...")
best_model = tf.keras.models.load_model("final_best_animal_classifier.keras")
loss, acc = best_model.evaluate(val_generator)
print(f"Final Validation Accuracy: {acc*100:.2f}%")


# 7. Grad-CAM Integration

def preprocess_image_for_model(image_path, target_size=(160, 160)):
    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy()

def save_and_display_gradcam(image_path, heatmap, alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite("gradcam_result.jpg", superimposed_img)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Visualization")
    plt.axis('off')
    plt.show()
# 8. Run Grad-CAM on an Image

image_path = "test_images/sample.jpg"  # Replace with your image path
img_array = preprocess_image_for_model(image_path)
heatmap, predicted_class = make_gradcam_heatmap(img_array, best_model)
print(f"Predicted Class: {class_names[predicted_class]}")
save_and_display_gradcam(image_path, heatmap)
