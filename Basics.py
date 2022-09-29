import cv2
import numpy as np
import face_recognition

# Converting to RGB
imgElon = face_recognition.load_image_file('ImagesBasic/ElonMusk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Torvalds.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Recognize and encode the faces
facLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]

# x,y positions, color, thickness
cv2.rectangle(imgElon, (facLoc[3], facLoc[0]), (facLoc[1], facLoc[2]), (0, 255, 0), 2)

# Recognize and encode the faces
facLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]

# x,y positions, color, thickness
cv2.rectangle(imgTest, (facLocTest[3], facLocTest[0]), (facLocTest[1], facLocTest[2]), (0, 255, 0), 2)

# compare faces through encodings, true on console for similar
results = face_recognition.compare_faces([encodeElon], encodeTest)

# Find the best match
faceDis = face_recognition.face_distance([encodeElon], encodeTest)

# Print the results
print(results, faceDis)

# Print the results
cv2.putText(imgTest, f'{results} {round(faceDis[0], 3)}', (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)