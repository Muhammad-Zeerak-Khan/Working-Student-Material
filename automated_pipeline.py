import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Setting up the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fixed Prompt Point
input_point = np.array([[118, 88]])
input_label = np.array([1])

# Define lists of values for contrast and brightness
contrast_values = [1.0, 1.2, 1.5, 1.8, 2.0]  # Avoid extremes like 0.0 or 3.0
brightness_values = [25, 50, 75, 100]  # Avoid negatives or very high values
sigma_values = [0.5, 1.0, 1.5, 2.0]  # Sigma values for the Gaussian blur
strength_values = [1.0, 1.5, 2.0, 2.5]  # Strength of the sharpening

# SAM2 Model Setup
sam2_checkpoint = "../checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Load the original image
image_path = 'images/Picture1.png'
image_name = os.path.basename(image_path)
image = Image.open(image_path)
image = np.array(image.convert("RGB"))

# Create main experimentation folder
experiment_folder = "experimentation_4"
os.makedirs(experiment_folder, exist_ok=True)

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['image_name', 'contrast_value', 'brightness_value', 'sigma_value', 'strength_value','segmentation_score'])

# Helper function to apply transformations
def apply_transformations(image, contrast, brightness):
    # Adjust contrast and brightness
    transformed_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return transformed_image

# Helper function for unsharp masking (sharpening)
def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

# Helper function to save image and results
def save_image_and_results(image, mask, hull_image, folder_path):
    cv2.imwrite(os.path.join(folder_path, "transformed_image.png"), image)
    cv2.imwrite(os.path.join(folder_path, "segmentation_mask.png"), mask)
    cv2.imwrite(os.path.join(folder_path, "final_hull_image.png"), hull_image)

# Function to calculate bounding box from the best mask
def calculate_rectangular_hull(mask, image):
    y_coords, x_coords = np.nonzero(mask)
    points = np.column_stack((x_coords, y_coords))
    rect = cv2.minAreaRect(points)
    box = np.intp(cv2.boxPoints(rect))
    image_with_box = image.copy()
    cv2.drawContours(image_with_box, [box], 0, (0, 255, 0), 2)
    return image_with_box

# Iterate over each combination of contrast, brightness, and sharpening parameters
for contrast in contrast_values:
    for brightness in brightness_values:
        for sigma in sigma_values:
            for strength in strength_values:
                print(f"Running experiment with contrast={contrast}, brightness={brightness}, sigma={sigma}, strength={strength}...")

                # Apply transformations
                transformed_image = apply_transformations(image, contrast, brightness)
                
                # Apply unsharp masking with current sigma and strength values
                sharpened_image = unsharp_mask(transformed_image, sigma=sigma, strength=strength)

                # Set the transformed image for SAM2 predictor
                predictor.set_image(sharpened_image)

                # Run prediction using the fixed point
                masks, scores, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )

                # Sort masks based on score and select the best one
                sorted_indices = np.argsort(scores)[::-1]
                best_mask = masks[sorted_indices[0]].astype(np.uint8)  # Select the best mask
                best_score = scores[sorted_indices[0]]  # Best score

                # Create a folder for this parameter combination
                folder_name = f"contrast_{contrast}_brightness_{brightness}_sigma_{sigma}_strength_{strength}"
                folder_path = os.path.join(experiment_folder, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Calculate bounding box from the best mask
                hull_image = calculate_rectangular_hull(best_mask, sharpened_image)

                # Save the transformed image, mask, and final bounding box image
                save_image_and_results(sharpened_image, best_mask * 255, hull_image, folder_path)

                # Store the results in the DataFrame using pd.concat() instead of append
                new_row = pd.DataFrame({
                    'image_name': [image_name],
                    'contrast_value': [contrast],
                    'brightness_value': [brightness],
                    'sigma_value': [sigma],
                    'strength_value': [strength],
                    'segmentation_score': [best_score]
                })

                results_df = pd.concat([results_df, new_row], ignore_index=True)

# Save the results DataFrame as a CSV file
results_df.to_csv(os.path.join(experiment_folder, 'experiment_results.csv'), index=False)
print("All experiments completed and results saved in 'experiment_results.csv'.")
