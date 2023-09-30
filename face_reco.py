import cv2
import csv
import numpy as np
import face_recognition

#x is image file
def compare_face(x):
    check = face_recognition.load_image_file(x)
    check = cv2.cvtColor(check, cv2.COLOR_BGR2RGB)
    check_encode = face_recognition.face_encodings(check)[0]
    
    with open("saved_encodings.csv", mode='r', newline ='') as file:
        reader = csv.reader(file)
        for row in reader:
            temp_encode = row[1]
            check_bool = face_recognition.compare_faces(temp_encode, check_encode)
            if check_bool == True:
                temp_name = row[0]
                check_result = [check_bool, temp_name,temp_encode]
                return check_result
            else:
                print("No match found")
#check_result is a boolean value
#can be used for future purposes
