import cv2
import numpy as np

class ImageTransformationTool:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)  # Load the image
        self.modified_image = self.original_image.copy()  # Keep a copy of the original image for modifications
        self.window_name = 'Image Transformation'

        # Initial trackbar values
        self.brightness = 0  # Ranges from 0 to 300
        self.contrast = 0    # Ranges from 0 to 100
        self.blur_level = 0  # Ranges from 0 to 10
        self.sharpen_intensity = 0  # Ranges from 0 to 100

        # Submit button attributes
        self.button_position = (10, 10, 150, 60)  # (x1, y1, x2, y2)
        self.button_color = (255, 255, 255)  # White background
        self.button_text_color = (0, 0, 0)  # Black text
        self.button_pressed_color = (200, 200, 200)  # Gray color when pressed

        # Create a flag to track if the submit button is clicked
        self.submitted = False

        # Create a window for trackbars and the image
        cv2.namedWindow(self.window_name)

        # Create trackbars
        self._create_trackbars()

    def _create_trackbars(self):
        """Create trackbars for brightness, contrast, blur, and sharpening adjustments."""
        cv2.createTrackbar('Brightness', self.window_name, self.brightness, 300, self._update_image)
        cv2.createTrackbar('Contrast', self.window_name, self.contrast, 100, self._update_image)
        cv2.createTrackbar('Blur Level', self.window_name, self.blur_level, 10, self._update_image)
        cv2.createTrackbar('Sharpen', self.window_name, self.sharpen_intensity, 100, self._update_image)

    def _draw_button(self, pressed=False):
        """Draw a better-looking submit button."""
        x1, y1, x2, y2 = self.button_position
        if pressed:
            cv2.rectangle(self.modified_image, (x1, y1), (x2, y2), self.button_pressed_color, -1, cv2.LINE_AA)  # Pressed color
        else:
            cv2.rectangle(self.modified_image, (x1, y1), (x2, y2), self.button_color, -1, cv2.LINE_AA)  # Normal color

        # Draw the button border with rounded corners
        cv2.rectangle(self.modified_image, (x1, y1), (x2, y2), (0, 0, 0), 2, cv2.LINE_AA)  # Black border

        # Add the text inside the button, centered
        text = "Submit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.8
        thickness = 2
        text_size, _ = cv2.getTextSize(text, font, text_scale, thickness)
        text_x = x1 + (x2 - x1 - text_size[0]) // 2  # Center the text horizontally
        text_y = y1 + (y2 - y1 + text_size[1]) // 2  # Center the text vertically

        cv2.putText(self.modified_image, text, (text_x, text_y), font, text_scale, self.button_text_color, thickness, cv2.LINE_AA)

    def _update_image(self, x):
        """Callback function to update the image based on trackbar values."""
        # Get current trackbar positions
        brightness_value = cv2.getTrackbarPos('Brightness', self.window_name)
        contrast_value = cv2.getTrackbarPos('Contrast', self.window_name)
        blur_value = cv2.getTrackbarPos('Blur Level', self.window_name)
        sharpen_value = cv2.getTrackbarPos('Sharpen', self.window_name)

        # Adjust brightness and contrast
        contrast_factor = 1.0 + (contrast_value / 100.0) * 2.0
        adjusted_image = cv2.convertScaleAbs(self.original_image, alpha=contrast_factor, beta=brightness_value)

        # Apply Gaussian blur
        kernel_size = blur_value * 2 + 1
        blurred_image = cv2.GaussianBlur(adjusted_image, (kernel_size, kernel_size), 0)

        # Apply sharpening filter
        sharpening_kernel = np.array([[ 0, -1,  0],
                                      [-1,  5, -1],
                                      [ 0, -1,  0]])
        sharpened_image = cv2.filter2D(blurred_image, -1, sharpening_kernel)
        # Blend sharpened image based on intensity
        self.modified_image = cv2.addWeighted(blurred_image, 1 - sharpen_value / 100.0, sharpened_image, sharpen_value / 100.0, 0)

        # Draw the "Submit" button
        self._draw_button()

        # Show the updated image
        cv2.imshow(self.window_name, self.modified_image)

    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function to handle 'Submit' button clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is inside the "Submit" button area
            x1, y1, x2, y2 = self.button_position
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Draw the pressed state of the button
                self._draw_button(pressed=True)
                cv2.imshow(self.window_name, self.modified_image)
                print("Submit button clicked. Final image modifications applied.")
                self.submitted = True  # Set the submitted flag to True

    def run(self):
        """Run the image transformation tool and handle user interactions."""
        # Set the mouse callback function for the window
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # Display the initial image
        self._update_image(0)

        # Main loop to keep the window open until 'q' is pressed or submit is clicked
        while True:
            key = cv2.waitKey(1) & 0xFF
            # If the user presses 'q', exit the loop
            if key == ord('q'):
                print("Quitting without submission.")
                break
            # If the submit button is clicked, exit the loop and return the modified image
            if self.submitted:
                break

        # Close the OpenCV window
        cv2.destroyAllWindows()
        # Return the final modified image
        return self.modified_image

# Testing Step 1: Create an instance of the class and run the tool
if __name__ == "__main__":
    image_tool = ImageTransformationTool('../sample.jpg')  # Replace with your image path
    final_image = image_tool.run()
    if final_image is not None:
        cv2.imwrite('final_transformed_image.png', final_image)
        print("Final transformed image saved as 'final_transformed_image.png'.")
