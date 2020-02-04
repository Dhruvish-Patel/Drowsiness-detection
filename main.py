"""
Created on Sun Sep 16 01:52:52 2018

@author: Dhruvish
"""

#!/usr/bin/python



import cv2
import dlib
import numpy as np
import time
import winsound



PREDICTOR_PATH = "D://study material//master-computer-vision-with-opencv-in-python//shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
#36 39 42 45


def right_eye(landmarks):
    r_eye_pts = []
    
    for i in range(36,42):
        r_eye_pts.append(landmarks[i])
        
    
    r_eye_all_pts = np.squeeze(np.asarray(r_eye_pts))
    
    return (r_eye_pts)


def left_eye(landmarks):
    l_eye_pts = []
    
    for i in range(42,48):
        l_eye_pts.append(landmarks[i])
        
    
    l_eye_all_pts = np.squeeze(np.asarray(l_eye_pts))
    
    return (l_eye_pts)


def eyes_close(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
    
        landmarks = get_landmarks(image)
        
        if landmarks == "error":
            return image, 0
        
        image_with_landmarks = annotate_landmarks(image, landmarks)
        
        r_eye = right_eye(landmarks)

        l_eye = left_eye(landmarks)
        
        a = np.linalg.norm(r_eye[1]-r_eye[5])
        b = np.linalg.norm(r_eye[2]-r_eye[4])
        c = np.linalg.norm(r_eye[0]-r_eye[3])
        
        r_ear = (a+b)/(2.0*c)


        a = np.linalg.norm(l_eye[1]-l_eye[5])
        b = np.linalg.norm(l_eye[2]-l_eye[4])
        c = np.linalg.norm(l_eye[0]-l_eye[3])
        
        l_ear = (a+b)/(2.0*c)
         
      
    
    
        return image_with_landmarks,r_ear,l_ear

    except:
        pass

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)


drows_status = False
drows = 0

if cap.isOpened():
        
    ret, frame = cap.read()
    
else:
    ret = False

while ret:
    
    ret, frame = cap.read()   
    image_landmarks, r_ear, l_ear = eyes_close(frame)
      
    prev_drows_status = drows_status
    
    if ((r_ear + l_ear)/2) < 0.2:

        while True:
            
            t_start = time.time()

            if (time.time()-t_start)>2:

                drows_status = True


                cv2.putText(frame, "Subject is SLEEPY!!", (50,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        
                cv2.putText(frame, "Warning!!!!!!", (150,150), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)

                winsound.Beep(1000, 5000)
    
    else:
        drows_status = False
        
    
    output_text = " Drowsiness Count: " + str(drows)

    cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)  
    
    if prev_drows_status == True and drows_status == False:
        drows += 1
        

    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Drowsiness Detection', frame )
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        
        #tkinter.messagebox.showinfo("CREATED BY-: ", "Raunak Jalan")

        break
        

cap.release()
cv2.destroyAllWindows() 
