import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import messagebox

# Paths to the YOLO model files
weights_path = "./models/yolov3.weights"
config_path = "./models/yolov3.cfg"
names_path = "./models/coco.names"

# List of classes to ignore (e.g., person)
ignored_classes = {"person"}

# Example product database with details and associations
product_database = {
    
    "apple": {"price": 1.00, "description": "Fresh apple", "stock": 100, "related": ["banana", "orange"]},
    "umbrella": {"price": 10.00, "description": "umbrella", "stock": 100, "related": ["rain coat"]},
    "sports ball": {"price": 20.00, "description": "ball", "stock": 100, "related": ["gloves", "bat"]},
    "bottle": {"price": 7.00, "description": "water bottle", "stock": 100, "related": ["cup", "glass"]},
    "lens sollution": {"price": 7.00, "description": "glasses", "stock": 100, "related": ["lens", "eyewear"]},
    "banana": {"price": 0.50, "description": "Ripe banana", "stock": 150, "related": ["apple", "orange"]},
    "orange": {"price": 0.75, "description": "Juicy orange", "stock": 200, "related": ["apple", "banana"]},
    "pen": {"price": 2.00, "description": "Ballpoint pen", "stock": 500, "related": ["notebook", "eraser"]},
    "notebook": {"price": 3.00, "description": "Spiral notebook", "stock": 300, "related": ["pen", "eraser"]},
    "eraser": {"price": 1.00, "description": "White eraser", "stock": 400, "related": ["pen", "notebook"]},
    "laptop": {"price": 1000.00, "description": "High-performance laptop", "stock": 50, "related": ["mouse", "keyboard"]},
    "mouse": {"price": 20.00, "description": "Wireless mouse", "stock": 100, "related": ["laptop", "keyboard"]},
    "keyboard": {"price": 30.00, "description": "Mechanical keyboard", "stock": 75, "related": ["laptop", "mouse"]},
    "shirt": {"price": 15.00, "description": "Casual shirt", "stock": 200, "related": ["jeans", "shoes"]},
    "jeans": {"price": 25.00, "description": "Blue jeans", "stock": 150, "related": ["shirt", "shoes"]},
    "shoes": {"price": 50.00, "description": "Running shoes", "stock": 100, "related": ["shirt", "jeans"]},
    "car": {"price": 250000000.00, "description": "Porsche car", "stock": 3, "related": ["wheels", "indoor spray"]},
    "wheels": {"price": 1000.00, "description": "Alloy wheels", "stock": 10, "related": ["car", "indoor spray"]},
    "indoor spray": {"price": 5.00, "description": "Air freshener spray", "stock": 100, "related": ["car", "wheels"]},
    "motorcycle": {"price": 1000000.00, "description": "Ducati motorcycle", "stock": 5, "related": ["gloves", "glasses"]},
    "gloves": {"price": 50.00, "description": "Leather gloves", "stock": 20, "related": ["motorcycle", "glasses"]},
    "glasses": {"price": 30.00, "description": "Sunglasses", "stock": 50, "related": ["motorcycle", "gloves"]},
    "bicycle": {"price": 500.00, "description": "Mountain bike", "stock": 10, "related": []},
    "chicken": {"price": 10.00, "description": "Fresh chicken", "stock": 50, "related": ["biryani masala", "curd", "spices"]},
    "curd": {"price": 3.00, "description": "Natural curd", "stock": 100, "related": ["biryani masala", "chicken", "spices"]},
    "spices": {"price": 2.50, "description": "Mixed spices", "stock": 300, "related": ["biryani masala", "chicken", "curd"]},
    "biryani masala": {"price": 5.00, "description": "Biryani masala spice mix", "stock": 200, "related": ["chicken", "curd", "spices"]},
    "tv": {"price": 500.00, "description": "Smart TV", "stock": 10, "related": ["remote", "soundbar"]},
    "remote": {"price": 10.00, "description": "TV remote control", "stock": 50, "related": ["tv", "soundbar"]},
    "soundbar": {"price": 100.00, "description": "Wireless soundbar", "stock": 20, "related": ["tv", "remote"]},
    "cell phone": {"price": 100.00, "description": "Smartphone", "stock": 20, "related": ["laptop"]}  # Updated key and description
}

# Load YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the COCO class names
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to detect objects in an image
def detect_objects(image):
    height, width, _ = image.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    object_positions = {}

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                label = str(classes[class_id])

                if label not in ignored_classes:  # Filter out ignored classes
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Track the object's center position
                    object_positions[label] = center_x

    # Apply non-maxima suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return object_positions, image

# Function to update the product database
def update_database(item_name):
    if item_name in product_database:
        product_database[item_name]['stock'] -= 1

# Function to get related products
def get_related_products(item_name):
    if item_name in product_database:
        related_items = product_database[item_name]['related']
        return [item for item in related_items if item in product_database]
    return []

# Function to display related products
def display_related_products():
    related_items = set()
    for item in purchased_items:
        related_items.update(get_related_products(item))
    
    if related_items:
        print("\n** Related Product Recommendations **")
        for item in related_items:
            details = product_database[item]
            print(f"{item}: ${details['price']} - {details['description']}")
        print("************************************")

# Function to handle adding items to the bill
def add_to_bill(item_name):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    response = messagebox.askyesno("Add to Bill", f"Add {item_name} to the bill?")
    root.destroy()  # Destroy the Tkinter instance
    if response:
        purchased_items.append(item_name)
        update_database(item_name)
    else:
        print(f"{item_name} was not added to the bill.")

# Function to detect significant movement
def detect_movement(prev_positions, curr_positions, threshold=50):
    for item, curr_pos_x in curr_positions.items():
        if item in prev_positions:
            prev_pos_x = prev_positions[item]
            # Calculate the horizontal movement distance
            distance = curr_pos_x - prev_pos_x
            if distance > threshold:  # Significant movement to the right
                return item
    return None

# Function to calculate the total bill
def calculate_total_bill():
    total = 0
    print("\n** Purchase Summary **")
    for item in purchased_items:
        price = product_database[item]["price"]
        print(f"{item}: ${price}")
        total += price
    print(f"Total Bill: ${total:.2f}")
    print("********\n")
    display_related_products()  # Display related product recommendations

# Function to handle the camera feed and object detection
def run_live_detection():
    cap = cv2.VideoCapture(0)
    prev_positions = {}
    already_detected = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_positions, output_frame = detect_objects(frame)
        cv2.imshow("Live Object Detection", output_frame)

        # Check for significant movement
        moving_item = detect_movement(prev_positions, curr_positions)
        if moving_item and moving_item not in already_detected:
            print(f"Detected movement of: {moving_item}")
            if moving_item not in ignored_classes:
                add_to_bill(moving_item)
                already_detected.add(moving_item)

        prev_positions = curr_positions  # Update previous positions

        if cv2.waitKey(1) & 0xFF == ord('q'):
            calculate_total_bill()
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to execute the program
if __name__ == "__main__":
    purchased_items = []  # Initialize an empty list for purchased items
    run_live_detection()
