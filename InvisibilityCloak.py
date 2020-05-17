''' TO write the video to a file kindly uncomment all multiline comments in the file'''

#Importing the required libraries
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
# To write the video to a file
'''
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')# note the lower case
fwidth = int(cap.get(3))
fheight = int(cap.get(4))
out = cv2.VideoWriter('InvisibilityCloak.mp4',fourcc , 10, (fwidth,fheight), True)
'''
time.sleep(3)
background = 0

# Capturing and Storing the Background Frame
for i in range(60):
    ret,background = cap.read()
background = np.flip(background,axis=1)

while(cap.isOpened()):
    ret, image = cap.read()
    if not ret:
        break
    image = np.flip(image,axis=1)
    
    # Converting from RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Creating masks with coordinates to detect the color BLUE
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    mask1 = cv2.inRange(hsv,lower_blue,upper_blue)

	'''# FOR RED COLORED MASK UNCOMMENT THIS CODE
    # Range for lower red
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)

    # Range for upper red
    lower_red = np.array([170,120,70]) #np.array([230,130,120])
    upper_red = np.array([180,255,255]) #np.array([235,210,90])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)

	#Adding the two red masks
    mask1 = mask1+mask2
    '''
    
    # Refining the created mask
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
 
    # Hiding away the masked part 
    mask2 = cv2.bitwise_not(mask1)

    # Generating the final output 
    res1 = cv2.bitwise_and(image,image,mask=mask2)
    res2 = cv2.bitwise_and(background, background, mask = mask1)
 
    output = cv2.addWeighted(res1,1,res2,1,0)
    
    # Writing the video in the file specified in the previous block
    '''out.write(output)'''
    
    # Generating live strean
    cv2.imshow("Invisibility Cloak",output)
    key = cv2.waitKey(1) & 0xFF
    # Press the key 'q' to stop streaming
    if key == ord("q"):
      break

cap.release()
'''out.release()'''
cv2.destroyAllWindows()