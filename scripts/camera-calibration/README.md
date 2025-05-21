# Camera Calibration

A comprehensive Python toolkit for camera calibration using OpenCV. This project provides tools to calibrate your camera, remove lens distortion, and apply the calibration to live video feeds.

## Overview

Camera calibration is the process of estimating the parameters of a camera's lens and image sensor. These parameters can be used to correct for lens distortion, measure the size of an object in world units, or determine the location of the camera in the scene.

This toolkit includes:

1. **Image Capture Tool**: Capture calibration images from your camera
2. **Calibration Tool**: Process the calibration images to compute camera parameters
3. **Live Undistortion**: Apply the calibration to a live video feed

## Requirements

- Python 3.6+
- OpenCV 4.5+
- NumPy 1.20+
- Matplotlib 3.4+ (for visualization)

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Capture Calibration Images

You need multiple images of a chessboard pattern from different angles and positions. The script `capture_calibration_images.py` helps you capture these images:

```bash
python capture_calibration_images.py
```

Controls:
- Press `c` to capture an image
- Press `q` or Escape to quit

The images will be saved in the `calibration_images` directory.

### Step 2: Run Camera Calibration

Process the calibration images to compute the camera matrix and distortion coefficients:

```bash
python camera_calibration.py
```

The calibration results will be saved in the `output` directory:
- `calibration_data.pkl`: Complete calibration data in pickle format
- `camera_matrix.txt`: Camera matrix in text format
- `distortion_coefficients.txt`: Distortion coefficients in text format
- Undistorted versions of the calibration images (if enabled)

### Step 3: Test the Calibration with Live Video

Apply the calibration to a live video feed:

```bash
python live_undistortion.py
```

Controls:
- Press `d` to toggle between distorted and undistorted view
- Press `q` to quit

## Configuration

All scripts use variables instead of command-line arguments for configuration. You can modify these variables at the top of each script:

### In `capture_calibration_images.py`:

```python
CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners per chessboard row and column
OUTPUT_DIRECTORY = 'calibration_images'  # Directory to save calibration images
```

### In `camera_calibration.py`:

```python
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners per chessboard row and column
SQUARE_SIZE = 2.5  # Size of a square in centimeters
CALIBRATION_IMAGES_PATH = 'calibration_images/*.jpg'  # Path to calibration images
OUTPUT_DIRECTORY = 'output'  # Directory to save calibration results
SAVE_UNDISTORTED = True  # Whether to save undistorted images
```

### In `live_undistortion.py`:

```python
CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)
CALIBRATION_FILE = 'output/calibration_data.pkl'  # Path to calibration data
```

## How It Works

### Camera Calibration Process

1. **Image Collection**: Capture multiple images of a chessboard pattern from different angles
2. **Corner Detection**: Detect the chessboard corners in each image
3. **Calibration**: Use the detected corners to compute the camera matrix and distortion coefficients
4. **Undistortion**: Apply the calibration to remove lens distortion from images

### Camera Model

The camera model used is the pinhole camera model with radial and tangential distortion:

- **Camera Matrix**: A 3x3 matrix containing the focal lengths and optical centers
- **Distortion Coefficients**: A vector containing the radial and tangential distortion coefficients

## Example Results

After calibration, you can expect:

1. **Undistorted Images**: Straight lines in the real world will appear straight in the images
2. **Accurate Measurements**: You can measure distances and sizes in the real world from the images
3. **3D Reconstruction**: You can use the calibration for 3D reconstruction or augmented reality applications

## Troubleshooting

### Common Issues

1. **Chessboard Not Detected**: Make sure the entire chessboard is visible in the image and well-lit
2. **Poor Calibration Results**: Use more images from different angles and positions
3. **Camera Not Found**: Check the CAMERA_ID parameter (usually 0 for built-in webcams)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for providing the computer vision algorithms
- The OpenCV documentation for the camera calibration tutorial