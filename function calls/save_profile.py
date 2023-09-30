import cv2
import os
import csv
import face_recognition

# Function to capture and save an image
def capture_profile_save(profile_name):
    folder_path = "lib\Saved_profiles\saved_profile_images"
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: in opening camera")
        return

    try:
        ret, frame = cam.read()

        if not ret:
            print("Error: could not capture image")
            return

        os.makedirs(folder_path, exist_ok=True)
        file_name = f"{profile_name}.jpg"  # Fix: Use f-string to insert the profile_name
        image_path = os.path.join(folder_path, file_name)

        cv2.imwrite(image_path, frame)
        print(f"Image saved at {folder_path} as {file_name}")

    finally:
        cam.release()
        cv2.destroyAllWindows()

    # Load the saved image using the file path
    img_bgr = face_recognition.load_image_file(image_path)

    # Convert to RGB format (required by face_recognition library)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cv2.imshow('bgr', img_bgr)
    cv2.imshow('rgb', img_rgb)
    cv2.waitKey(0)

    # Recognize the face in the image and save the encoding
    fixed_encode = face_recognition.face_encodings(img_rgb)[0]

    # Save the encoding in a CSV file
    csv_file = "saved_encodings.csv"
    with open(csv_file, mode='a', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([profile_name, fixed_encode])

# Call the function to capture and save the image
#test command for particular file mentioned below
#capture_profile_save("Abhay")
