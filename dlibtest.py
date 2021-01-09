import cv2.cv2 as cv2 
import dlib 
import numpy as np
import math
from enum import Enum

class Location(Enum):
    cheek = "cheek"
    eye = "eye"

location = Location.cheek
overlay = cv2.imread("fire.png", cv2.IMREAD_UNCHANGED)

def put4ChannelImageOn4ChannelImage(back, fore, x, y):
    rows, cols, channels = fore.shape    
    trans_indices = fore[...,3] != 0 
    overlay_copy = back[y:y+rows, x:x+cols] 
    overlay_copy[trans_indices] = fore[trans_indices]
    back[y:y+rows, x:x+cols] = overlay_copy


def getScale(faceCoords, overlayImg, location):
    '''
    Returns scalar value for overlay image
    '''
    scalar = 1
    if location == Location.cheek:
        scalar = (abs(faceCoords.part(15).x - faceCoords.part(1).x)/4) / overlayImg.shape[0]
    elif location == Location.eye:
        scalar = (abs(faceCoords.part(39).x - faceCoords.part(36).x)) * 1.5 / overlayImg.shape[0]
    return scalar


def getFaceAngle(faceCoords):
    ''' 
    Return angle to rotate overlay by in degrees
    '''
    #27 - 8
    #cot y/x
    try:
        angle = math.atan((faceCoords.part(27).x - faceCoords.part(8).x) / (faceCoords.part(27).y - faceCoords.part(8).y))
        return np.rad2deg(angle)
    except:
        return 0


def getCoords(faceCoords, overlay, location):
    '''
    Return left and right points for overlay depending on location
    '''
    coords = {}
    if location == Location.cheek:
        leftX = int(((landmarks.part(31).x - landmarks.part(2).x)/4 + landmarks.part(2).x) - (overlay.shape[0]/2))
        leftY = int(((landmarks.part(48).y - landmarks.part(41).y)/2 + landmarks.part(41).y) - (overlay.shape[1]/2))
        rightX = int(((landmarks.part(14).x - landmarks.part(35).x)*3/4 + landmarks.part(35).x) - (overlayEdit.shape[0]/2))
        rightY = int(((landmarks.part(54).y - landmarks.part(45).y)/2 + landmarks.part(45).y) - (overlay.shape[1]/2))

        coords['left'] = (leftX,leftY)
        coords['right'] = (rightX,rightY)
    elif location == Location.eye:
        leftX = int(((landmarks.part(39).x - landmarks.part(36).x)/2 + landmarks.part(36).x) - (overlay.shape[0]/2))
        leftY = int(((landmarks.part(40).y - landmarks.part(38).y)/2 + landmarks.part(38).y) - (overlay.shape[1]/2))
        rightX = int(((landmarks.part(45).x - landmarks.part(42).x)/2 + landmarks.part(42).x) - (overlayEdit.shape[0]/2))
        rightY = int(((landmarks.part(47).y - landmarks.part(43).y)/2 + landmarks.part(43).y) - (overlay.shape[1]/2))

        coords['left'] = (leftX,leftY)
        coords['right'] = (rightX,rightY)
    return coords

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat')

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
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 
        img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        #scale overlay
        scale = getScale(landmarks, overlay, location)
        overlayEdit = cv2.resize(overlay,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        
        #rotate overlay
        (h, w) = overlayEdit.shape[:2]
        center = (w / 2, h / 2)
        angle = getFaceAngle(landmarks)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        overlayEdit = cv2.warpAffine(overlayEdit, M, (w, h))

        #transform overlay
        overlayCoords = getCoords(landmarks, overlayEdit, location)
        put4ChannelImageOn4ChannelImage(img_BGRA, overlayEdit, overlayCoords['left'][0], overlayCoords['left'][1])
        put4ChannelImageOn4ChannelImage(img_BGRA, overlayEdit, overlayCoords['right'][0], overlayCoords['right'][1])

    cv2.imshow(winname="Face", mat=img_BGRA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
