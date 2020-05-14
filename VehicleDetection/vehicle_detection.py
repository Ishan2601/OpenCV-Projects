"""Vehicle Detection in Videos"""

# Importing OpenCV library
import cv2

#Getting the cascade and video source
cascade_src = 'cars.xml'
video_src = 'dataset/video1.avi'
#video_src = 'dataset/video2.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    #Close if video is completed
    if (type(img) == type(None)):
        break
    #Converting to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Applying Cars Cascade
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    #Spotting out cars in video
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27: #Esc key to stop
        break

cv2.destroyAllWindows()