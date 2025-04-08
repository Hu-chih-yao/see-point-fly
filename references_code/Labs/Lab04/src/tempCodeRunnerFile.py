import cv2

# Step 1: Initialize the webcam. The 0 denotes the built-in webcam.
cap = cv2.VideoCapture(1)

# Step 2: Check if the camera opened successfully.
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Step 3: Loop to continuously capture frames and display them.
while True:
    # Read a new frame from the camera.
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the captured frame.
    cv2.imshow('Camera Feed', frame)

    # Close the window and stop capturing when 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 4: Release the camera and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
