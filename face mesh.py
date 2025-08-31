import cv2 as cv
import mediapipe as mp

import time

myfacemesh = mp.solutions.face_mesh
facemesh = myfacemesh.FaceMesh(max_num_faces=1)
mpDraw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)
ctime = 0
ptime = 0
while True:
    success, img = cap.read()
    if success:
        imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = facemesh.process(imgrgb)
        if results.multi_face_landmarks:
            for facelms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, facelms,myfacemesh.FACEMESH_TESSELATION)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        cv.putText(img, str(int(fps)), (30, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        ptime = ctime
    cv.imshow('Video', img)
    cv.waitKey(1)