

import cv2

f_casc=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ey_casc=cv2.CascadeClassifier('haarcascade_eye.xml')
smile_casc=cv2.CascadeClassifier('haarcascade_smile.xml')
def detect(gray,frame):
    faces= f_casc.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_g=gray[y:y+h, x:x+w]
        roi_c=frame[y:y+h, x:x+w]
        eyes=ey_casc.detectMultiScale(roi_g,1.2,5)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_c, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)
            
        smile=smile_casc.detectMultiScale(roi_g,1.8,25)
        for(sx,sy,sw,sh) in smile:
            cv2.rectangle(roi_c, (sx,sy), (sx+sw,sy+sh),(0,255,255),2)
    return frame
    
    
v_capt = cv2.VideoCapture(0) # We turn the webcam on.

while True: # We repeat infinitely (until break):
    _, frame = v_capt.read() # We get the last frame.
    gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    canv = detect(gry,frame) # We get the output of our detect function.
    cv2.imshow('Video', canv) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

v_capt.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.
