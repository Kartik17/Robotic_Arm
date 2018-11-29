import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_before', gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    gray = clahe.apply(gray)


    #gray = cv2.equalizeHist(gray)
    cv2.imshow('gray_after', gray)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2)
        rects_eye = np.array([[x,y,x+w,y+h] for (x,y,w,h) in eyes])
        picked_rect_eye = non_max_suppression(rects_eye , overlapThresh=0.80)

        for (ex,ey,exb,eyb) in picked_rect_eye:
            cv2.rectangle(roi_color,(ex,ey),(exb,eyb),(0,255,0),2)

    cv2.imshow('img',img)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()