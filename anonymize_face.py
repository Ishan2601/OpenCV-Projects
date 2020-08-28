# Blur and Anonymize faces

# Importing the necessary libraries
import cv2
import numpy as np

# Defining the haar cascade classifier
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read and resize the input image
im = cv2.imread('img3.jpg')
small = cv2.resize(im,(480,640),cv2.INTER_AREA)


def anonymize(img,method,scale=3):
	image = img.copy()
	# Convert image to grayscale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
	face = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
	# If face not found return None
	if not len(face):
		return None,None
	# Define ROI for the face 
	for (x,y,w,h) in face:
		roi = image[y:y+h,x:x+w]
	# Call the blurring method which user asks for
	if method.lower()[0] == 'b':
		blurred = blur_face(roi,scale)
	else:
		blurred = pixelate_face(roi,scale)
	# Replace the face in image with the blurred face and return the image
	image[y:y+h,x:x+w] = blurred
	return image

def blur_face(image, factor=3.0):
	# Get Image height and width
	(h, w) = image.shape[:2]
	# Calculate kernel width and height
	kW = int(w / factor)
	kH = int(h / factor)
	# Make sure that the kernel width and height are odd
	if kW % 2 == 0:
		kW -= 1
	
	if kH % 2 == 0:
		kH -= 1
	
	return cv2.GaussianBlur(image, (kW, kH), 0)

def pixelate_face(image, blocks=3):
	# Get Image height and width
	(h, w) = image.shape[:2]
	# Get the values of x and y where we want to step
	xstep = np.linspace(0, w, blocks + 1, dtype="int")
	ystep = np.linspace(0, h, blocks + 1, dtype="int")
	#Loop through the steps
	for i in range(1, len(ystep)):
		for j in range(1, len(xstep)):
			# For each block, get the values of x and y between which we have to calculate mean pixel value
			startX = xstep[j - 1]
			startY = ystep[i - 1]
			endX = xstep[j]
			endY = ystep[i]
			# Compute the roi
			roi = image[startY:endY, startX:endX]
			# Calculate the mean pixel value
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			# Apply it to the block
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)

	return image

# Call the anonymize function
anonymous = anonymize(small,'p',10)

# Run an infinite loop
while True:
	# Display original and blurred images
	cv2.imshow('Original',small)
	cv2.imshow('Anonymized',anonymous)
	# If 'Q' key is pressed exit the loop
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
# Close all open windows.
cv2.destroyAllWindows()