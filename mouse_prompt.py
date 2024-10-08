import cv2

class PromptPointCollector:
    def __init__(self, image):
        """
        Initialize the PromptPointCollector class.
        
        Args:
            image (np.array): Transformed image from Step 1.
        """
        self.image = image.copy()  # Make a copy of the image to display
        self.points = []  # List to store prompt points (x, y)
        self.window_name = 'Prompt Point Collector'

        # Create a window to display the image
        cv2.namedWindow(self.window_name)

    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function to collect prompt points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the clicked point (x, y) to the points list
            self.points.append((x, y))
            print(f"Point added at: ({x}, {y})")

            # Draw a small circle on the image at the clicked point
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)  # Green circle
            cv2.imshow(self.window_name, self.image)

    def collect_points(self):
        """
        Display the image and collect points via mouse clicks.
        
        Returns:
            list: List of (x, y) coordinates representing the prompt points.
        """
        # Set the mouse callback function to capture points
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # Display the image and wait for the user to finish (press 'q' to quit)
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF

            # Break the loop if 'q' is pressed
            if key == ord('q'):
                break

        # Close the window
        cv2.destroyAllWindows()
        # Return the list of collected points
        return self.points

# Testing Step 2: Create an instance of the class and collect points on the modified image
if __name__ == "__main__":
    # Load the final transformed image from Step 1 (replace with actual path or variable)
    transformed_image = cv2.imread('final_transformed_image.png')  # Example path
    
    # Create an instance of PromptPointCollector with the transformed image
    point_collector = PromptPointCollector(transformed_image)
    
    # Collect points by clicking on the image and save the collected points
    prompt_points = point_collector.collect_points()
    
    # Output the collected prompt points
    print(f"Collected prompt points: {prompt_points}")
