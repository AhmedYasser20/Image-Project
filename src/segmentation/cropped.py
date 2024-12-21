import cv2
from PIL import Image

# Load the image
image_path = "data\input\test1_clean_clean.jpg"  # Replace with your image path
original_image = cv2.imread(image_path)

# Step 1: Convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply edge detection
edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

# Step 3: Find horizontal lines using contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: Sort contours by vertical position
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
bounding_boxes = sorted(bounding_boxes, key=lambda box: box[1])  # Sort by y-coordinate

# Step 5: Crop each detected line
cropped_lines = []
for box in bounding_boxes:
    x, y, w, h = box
    # Ignore very small contours (noise)
    if h > 10:  # Minimum height to consider a line
        cropped_line = original_image[y:y + h, x:x + w]
        cropped_lines.append(cropped_line)

# Step 6: Resize each cropped line
fixed_size = (800, 200)  # Desired width and height
resized_lines = [cv2.resize(line, fixed_size) for line in cropped_lines]

# Step 7: Save each cropped and resized line
for idx, resized_line in enumerate(resized_lines):
    output_path = f"cropped_line_{idx + 1}.png"
    cv2.imwrite(output_path, resized_line)
    print(f"Saved: {output_path}")
