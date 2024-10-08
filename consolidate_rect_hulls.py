import os
import shutil

# Define the root directory containing all experiment folders
experiment_root_folder = "experimentation_4"
hull_output_folder = os.path.join(experiment_root_folder, "rectangular_hulls")

# Create the `rectangular_hulls` folder if it doesn't exist
os.makedirs(hull_output_folder, exist_ok=True)

# Iterate through each subfolder in the `experimentation_3` folder
for folder_name in os.listdir(experiment_root_folder):
    # Skip the "rectangular_hulls" folder itself if it already exists
    if folder_name == "rectangular_hulls":
        continue

    # Construct the full path to the current folder
    folder_path = os.path.join(experiment_root_folder, folder_name)

    # Check if the folder path is a directory (to avoid processing files)
    if os.path.isdir(folder_path):
        # Look for the `final_hull_image.png` file inside this folder
        hull_image_path = os.path.join(folder_path, "final_hull_image.png")

        # Check if the `final_hull_image.png` file exists in the current folder
        if os.path.exists(hull_image_path):
            # Construct the new file name for the hull image using the folder name
            new_hull_image_name = f"{folder_name}.png"

            # Define the destination path in the `rectangular_hulls` folder
            destination_path = os.path.join(hull_output_folder, new_hull_image_name)

            # Copy the hull image to the destination folder with the new name
            shutil.copy(hull_image_path, destination_path)
            print(f"Copied {hull_image_path} to {destination_path}")

print("All rectangular hulls have been copied to the `rectangular_hulls` folder.")
