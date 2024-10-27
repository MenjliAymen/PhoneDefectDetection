import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. U-Net Model Definition
def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    # Encoding / Downsampling path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    # Decoding / Upsampling path
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    # Output layer with sigmoid activation
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Instantiate the U-Net model
model = unet_model()
model.summary()

# 2. Data Preprocessing and Loading
# Function to convert yellow lines in masks to binary (white)
def convert_yellow_to_binary(mask, lower_yellow=(20, 100, 100), upper_yellow=(30, 255, 255)):
    hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    yellow_mask = cv2.inRange(hsv_mask, lower_yellow, upper_yellow)  # Detect yellow
    binary_mask = (yellow_mask > 0).astype(np.uint8)  # Convert to binary (0 or 1)
    return binary_mask

# Load dataset
def load_data(image_dir, mask_dir, img_size=(128, 128)):
    images = []
    masks = []
    image_names = os.listdir(image_dir)

    for image_name in image_names:
        # Load and preprocess image
        img = cv2.imread(os.path.join(image_dir, image_name))
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize to [0, 1]
        images.append(img)

        # Load and preprocess corresponding mask
        mask_name = image_name
        mask = cv2.imread(os.path.join(mask_dir, mask_name))
        mask = convert_yellow_to_binary(mask)  # Convert yellow lines to binary
        mask = cv2.resize(mask, img_size)
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        masks.append(mask)

    return np.array(images), np.array(masks)

# Paths to images and masks
image_dir = 'Images/image/'
mask_dir = 'Images/mask/'

# Load the images and masks
images, masks = load_data(image_dir, mask_dir)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

data_gen_args = dict(rotation_range=10,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Create generators
seed = 42
image_generator = image_datagen.flow(X_train, batch_size=16, seed=seed)
mask_generator = mask_datagen.flow(y_train, batch_size=16, seed=seed)

# Custom generator to yield images and masks together

def train_generator(image_gen, mask_gen):
    while True:
        X_batch = next(image_gen)  # Use `next()` to get the next batch
        Y_batch = next(mask_gen)   # Use `next()` to get the next batch of masks
        yield X_batch, Y_batch

# 4. Model Training
epochs = 20
train_gen = train_generator(image_generator, mask_generator)

history = model.fit(train_gen,
                    steps_per_epoch=len(X_train) // 16,
                    epochs=epochs,
                    validation_data=(X_val, y_val))

# 5. Plot Training History
def plot_training(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot the training history
plot_training(history)

# Save the trained model
model.save('mobile_phone_defect_model_unet4.h5')
