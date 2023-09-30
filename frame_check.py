import cv2
import os
import datetime
from face_reco import compare_face

def frame_check():
    cam = cv2.VideoCapture(0)

    time = datetime.datetime.now().strftime("%d/%m/%y/%H/%M/%S")
    if not cam.isOpened():
            print("Error: in opening camera")
            return
    folder_path = "cache\captures"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    known_faces =[]

    while True:
        try:
            ret, frame = cam.read()

            if not ret:
                print("Error: could not capture image")
                return

            os.makedirs(folder_path, exist_ok=True)
            file_name = f"{time}.jpg" 
            image_path = os.path.join(folder_path, file_name)

            cv2.imwrite(image_path, frame)
            print(f"Image saved at {folder_path} as {file_name}")
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1,5, minSize=(30,30))

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                face_center = (x + w//2, y + h//2)
                new_face = True
                for(know_x,know_y) in known_faces:
                    distance = ((know_x - face_center[0])**2 + (know_y - face_center[1])**2)**0.5
                    if distance < 50:
                        new_face = False
                        break
                if new_face:
                    print("New face detected")
                    known_faces.append(face_center)
                    
                    newface_image_name = f"newface_{time}.jpg"
                    newface_image_path = f"cache\unsaved_faces\{newface_image_name}"
                    cv2.imwrite(newface_image_path, frame[y:y+h, x:x+w])
                    print(f"New face image saved at {newface_image_path} as {newface_image_name}")
                    
                    compare_face(newface_image_path)
                    #returns check_result
                    #can be used for further processess
        finally:
            cam.release()
            cv2.destroyAllWindows()