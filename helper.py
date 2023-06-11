import cv2

# Load the image using OpenCV
img = cv2.imread('image.jpg')

# Resize the image to 32x32
resized_img = cv2.resize(img, (32, 32))

# Save the resized image
cv2.imwrite('resized_image.jpg', resized_img)
