# Smart Shopping Cart MVP Prototype

## Overview
The Smart Shopping Cart MVP is a proof-of-concept project designed to enhance the shopping experience through automated product detection. This prototype uses YOLOv5 for object detection, running entirely in the terminal. It detects items placed in the cart and displays the bill, streamlining the checkout process.

## Features
- **Object Recognition:** Uses YOLOv5 to detect and identify products in the shopping cart.
- **Command-Line Interface:** Displays detected items and calculates the total bill directly in the terminal.
- **Basic Product Recommendations:** Provides simple suggestions based on detected items (to be expanded in future versions).

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- YOLOv5 (pre-trained weights)

### Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/Smart-shopping-cart.git
    cd smart-shopping-cart
    ```

2. **Set Up Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. **Download YOLOv5 Weights:**
    - Download the YOLOv5 pre-trained weights from the [YOLOv5 repository](https://github.com/ultralytics/yolov5) and place them in the `weights` directory.

2. **Run the Object Detection Script:**
    - Start the detection process by running the following command:
    ```bash
    python Item_recognition.py
    ```

3. **Displaying the Bill:**
    - Detected items will be displayed in the terminal along with their prices.
    - The script will calculate and display the total bill.

### Usage

- **Object Detection:** Place items in front of the camera or use images/videos. The system will detect and list the items.
- **Bill Calculation:** Automatically sums up the prices of detected items and displays the total bill in the terminal.

## Next Steps

- **Integration with Vertex AI:** To be developed in future versions for scalable model training and deployment.
- **Frontend Interface:** A graphical interface will be implemented to enhance user interaction and experience.
- **Improved Recommendations:** Advanced algorithms will be added for personalized product recommendations.

## Contributing

Feel free to fork this repository, create a feature branch, and submit a pull request for new features or bug fixes.

