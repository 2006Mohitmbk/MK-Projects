import cv2

# Start webcam
camera = cv2.VideoCapture(0)

# Capture first frame
ret, previous_frame = camera.read()

if not ret:
    print("Error: Could not access webcam.")
    exit()

# Convert first frame to grayscale
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Blur the image
previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

while True:

    # Capture current frame
    ret, current_frame = camera.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert current frame to grayscale
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Blur current frame
    current_gray = cv2.GaussianBlur(
        current_gray,
        (21, 21),
        0
    )

    # Find difference between frames
    difference = cv2.absdiff(previous_gray, current_gray)

    # Convert difference into binary image
    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # Find moving areas
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        # Ignore very small movements
        area = cv2.contourArea(contour)

        if area < 500:
            continue

        # Get rectangle around movement
        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle
        cv2.rectangle(
            current_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            current_frame,
            "Motion Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Display result
    cv2.imshow("Motion Detection", current_frame)

    # Update previous frame
    previous_gray = current_gray

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()