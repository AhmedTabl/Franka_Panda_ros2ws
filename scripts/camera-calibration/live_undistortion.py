import cv2
import numpy as np
import os
import pickle

# Live undistortion parameters
CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)
CALIBRATION_FILE = 'output/calibration_data.pkl'  # Path to calibration data

def live_undistortion():
    """
    Demonstrate live camera undistortion using calibration results.
    """
    # Check if calibration file exists
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Error: Calibration file not found at {CALIBRATION_FILE}")
        print("Please run camera_calibration.py first to generate calibration data.")
        return
    
    # Load calibration data
    with open(CALIBRATION_FILE, 'rb') as f:
        calibration_data = pickle.load(f)
    
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['distortion_coefficients']
    
    print("Loaded camera calibration data:")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients: {dist.ravel()}")
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_ID}")
        return
    
    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    # Calculate optimal camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    
    # Create undistortion maps
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), 5)
    
    print("Press 'q' to quit, 'd' to toggle distortion correction")
    
    # Flag to toggle distortion correction
    correct_distortion = True
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        if correct_distortion:
            # Apply undistortion
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            
            # Crop the image (optional)
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
            
            # Resize to original size for display
            undistorted = cv2.resize(undistorted, (width, height))
            
            # Add text to indicate undistorted view
            cv2.putText(undistorted, "Undistorted", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the undistorted frame
            cv2.imshow('Camera Feed', undistorted)
        else:
            # Add text to indicate original view
            cv2.putText(frame, "Original", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the original frame
            cv2.imshow('Camera Feed', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            break
        
        # 'd' to toggle distortion correction
        elif key == ord('d'):
            correct_distortion = not correct_distortion
            print(f"Distortion correction {'ON' if correct_distortion else 'OFF'}")
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_undistortion()