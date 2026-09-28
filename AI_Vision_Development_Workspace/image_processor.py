import cv2

# Step 1: Read the image
image = cv2.imread("images/sample.jpg")

# Check whether image is loaded
if image is None:
    print("Error: Image not found.")
    exit()

# Step 2: Display orignal image information
height, width, channels = image.shape

print("Image Width:", width)
print("Image Height:", height)
print("Number of Channels:", channels)

# Step 3: Display original image
cv2.imshow("Orignal image", image)

# Step 4: Convert image to greyscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Greyscale Image", gray_image)

# Step 5: Resize image
resized_image = cv2.resize(image, (400, 300))

cv2.imshow("Resized Image", resized_image)

# Step 6: Save grayscale image
cv2.imwrite("output/grayimage.jpg", gray_image)

print("Grayscale image saved successfully.")

# wait for keyboard input
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()