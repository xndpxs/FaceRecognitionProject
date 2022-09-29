import cv2
import numpy as np
import face_recognition
import os

# Grabbing the images from the folder
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:    # Bucle to grab names
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])      # grab the name without .jpg
print(classNames)

# Function to find the encoders
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# call the function
encodeListKnown = findEncodings(images)
print('Encoding Complete')


# Capture from webcam

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)            # resizing image to small 1/4
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurrentFrame = face_recognition.face_locations(imgS)           # encoding the faces
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)   # check the min, min more exact match
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)






