# Face Recognition with Gender and Age prediction

import cv2
import pafy

url = 'https://www.youtube.com/watch?v=SvldnZ6qMGU'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4") 
 
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # Load the cascade for the eyes.

def detect(gray, frame): # Function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
        
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Apply the detectMultiScale method from the face cascade to locate
    #one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) 
        
        # Placing a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # Apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes,
            #but inside the referential of the face.
        
        face_img = frame[y:y+h, h:h+w].copy()
        
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False, crop=True)
        
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i=gender_preds[0].argmax()
        gender = gender_list[gender_preds[0].argmax()]
        gender_conf=gender_preds[0][i]
        
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        j=age_preds[0].argmax()
        age_conf=age_preds[0][j]
        
        
        #overlay_text = "%s %d" % (gender, gender_conf)
        overlay_text_g = "Gender: {} -> {:.2f}%".format(gender, gender_conf * 100)
        overlay_text_a = "Age: {} -> {:.2f}%".format(age, age_conf * 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, overlay_text_g, (x, y), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, overlay_text_a, (x, y+h), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    return frame # Return the image with the detector rectangles.

video_capture = cv2.VideoCapture(play.url) # Webcam on

while True: 
    _, frame = video_capture.read()
    #FOR INPUT IMAGE cv2.imread(r'C:\Users\LAPTOP\Desktop\images_male_p\p\i445xa-mg.jpg') # We get the last frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    canvas = detect(gray, frame) # Output of our detect function.
    cv2.imshow('Youtube Video', canvas) 
    if cv2.waitKey(1) & 0xFF == ord('q'): # to break loop
        break 


video_capture.release() 
cv2.destroyAllWindows() 