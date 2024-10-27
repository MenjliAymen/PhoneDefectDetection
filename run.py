import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.src.saving import load_model
from scipy.ndimage import label

# Load the trained U-Net model
model = load_model("mobile_phone_defect_model_unet4.h5")

# Function to preprocess the input image
def preprocess_image(image_path, size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 128, 128, 3)
    return img

# Function to postprocess the mask
def postprocess_mask(mask, original_size):
    mask = mask[0, :, :, 0]  # Remove batch and channel dimensions
    mask = cv2.resize(mask, original_size)  # Resize back to original image size
    return mask

# Function to adjust contrast if no scratches are found
def increase_contrast(image, alpha=1.5, beta=0):
    """
    Increases the contrast of the input image by scaling pixel values using:
    new_image = alpha * original_image + beta
    """
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

# Function to predict defects on the screen
def predict_defect(image_path, use_contrast=False):
    # Load original image for displaying
    original_img = cv2.imread(image_path)
    original_size = (original_img.shape[1], original_img.shape[0])  # Get original image size

    # If use_contrast is True, increase contrast of the image
    if use_contrast:
        original_img = increase_contrast(original_img)

    # Preprocess the input image
    img = cv2.resize(original_img, (128, 128))  # Resize for the model input
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 128, 128, 3)

    # Predict the mask using the model
    pred_mask = model.predict(img)

    # Postprocess the predicted mask
    pred_mask = postprocess_mask(pred_mask, original_size)

    return original_img, pred_mask

# Function to display original image and mask with bounding boxes and probabilities
def display_result_with_boxes(original_img, mask):
    # Threshold the mask to get binary mask
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Find connected components in the mask
    labeled_mask, num_features = label(binary_mask)

    if num_features == 0:
        print("No scratches detected.")
    else:
        # Loop over each detected region
        for i in range(1, num_features + 1):
            # Get the region corresponding to the current label
            region = (labeled_mask == i).astype(np.uint8)

            # Find the bounding box of the region
            x, y, w, h = cv2.boundingRect(region)

            # Extract the mask values within the region to calculate the probability
            region_mask_values = mask[y:y+h, x:x+w]
            confidence_score = np.mean(region_mask_values)

            # Draw a bounding box around the detected scratch
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the probability (confidence score) near the bounding box
            cv2.putText(original_img, f"{confidence_score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the original image with bounding boxes and probabilities
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Scratch Detection with Bounding Boxes and Confidence Scores")
    plt.show()

# Example usage:
image_path = "Images/image/p6.jpg"  # Path to the new input image

# First pass: predict defects without contrast adjustment
original_img, pred_mask = predict_defect(image_path)

# Check if scratches are detected
if np.sum(pred_mask) == 0:
    print("No scratches found on first pass. Increasing contrast...")
    # Second pass: predict defects with contrast adjustment
    original_img, pred_mask = predict_defect(image_path, use_contrast=True)

# Display result with bounding boxes and probabilities
display_result_with_boxes(original_img, pred_mask)
