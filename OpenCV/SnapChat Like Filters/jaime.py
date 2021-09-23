import cv2
import numpy as np
import matplotlib.pyplot as plt

classifier_eyes=cv2.CascadeClassifier('/Users/parthsharma/Desktop/Data Science CB/Train/third-party/frontalEyes35x16.xml')
classifier_nose=cv2.CascadeClassifier('/Users/parthsharma/Desktop/Data Science CB/Train/third-party/Nose18x15.xml')

img_before=cv2.imread('/Users/parthsharma/Desktop/Data Science CB/Train/Jamie_Before.jpg')
gray=cv2.cvtColor(img_before,cv2.COLOR_BGR2GRAY)

eyes=classifier_eyes.detectMultiScale(gray,1,3,5)

for (x,y,w,h) in eyes:
    cv2.rectangle(img_before,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("frame",img_before)
cv2.waitKey(0)
cv2.destroyAllWindows()