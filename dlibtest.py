import cv2.cv2 as cv2 
import dlib 
import numpy as np
import math

def put4ChannelImageOn4ChannelImage(back, fore, x, y):
    rows, cols, channels = fore.shape    
    trans_indices = fore[...,3] != 0 
    overlay_copy = back[y:y+rows, x:x+cols] 
    overlay_copy[trans_indices] = fore[trans_indices]
    back[y:y+rows, x:x+cols] = overlay_copy

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat')

overlay = cv2.imread("test4.png", cv2.IMREAD_UNCHANGED)

while(True):
    ret, img = cap.read()
    img_BGRA = img

    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(image=gray, box=face)
        x = landmarks.part(27).x
        y = landmarks.part(27).y    

        b_channel, g_channel, r_channel = cv2.split(img)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        scale = (abs(landmarks.part(15).x - landmarks.part(1).x)/4) / overlay.shape[0]
        overlayResize = cv2.resize(overlay,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        
        offset_x_left = int(((landmarks.part(31).x - landmarks.part(2).x)/4 + landmarks.part(2).x) - (overlayResize.shape[0]/2))
        offset_x_right = int(((landmarks.part(14).x - landmarks.part(35).x)*3/4 + landmarks.part(35).x) - (overlayResize.shape[0]/2))
        offset_y = int(((landmarks.part(48).y - landmarks.part(41).y)/2 + landmarks.part(41).y) - (overlayResize.shape[1]/2))

        put4ChannelImageOn4ChannelImage(img_BGRA, overlayResize, offset_x_left, offset_y)
        put4ChannelImageOn4ChannelImage(img_BGRA, overlayResize, offset_x_right, offset_y)

    cv2.imshow(winname="Face", mat=img_BGRA)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
