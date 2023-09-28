import cv2
import numpy as np
import face_recognition

#we consider image taken saved as image.jpg and the image to be compared with as original.jpg

img_bgr = face_recognition.load_image_file('lib/original.jpg')
#since face_recognition takes only bgr
#we convert it to rgbw
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
cv2.imshow('bgr',img_bgr)
cv2.imshow('rgb',img_rgb)
cv2.waitKey(0)

#recognize the face in the image and save the encoding
fixed_encode = face_recognition.face_encodings(img_rgb)[0]

check = face_recognition.load_image_file('cache/captures/image.jpg')
check = cv2.cvtColor(check, cv2.COLOR_BGR2RGB)
check_encode = face_recognition.face_encodings(check)[0]
check_result = face_recognition.compare_faces([fixed_encode], check_encode)

#check_result is a boolean value
#can be used for future purposes
