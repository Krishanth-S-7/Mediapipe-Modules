import cv2 as cv
import time
import Hand_tracking_module as htm
import body_detection_module as bd
import numpy as np
import math
cap = cv.VideoCapture(0)
mybody = bd.BodyDetector(min_det_con=0.9)
detector = htm.HandDetector(number_hands=1,det_con=0.7)
ctime = 0
ptime = 0
prevangle = 180
up = False
minangle = 0
pushupcount = 0
first = True
first1 = True
start = False
countdown = 5
ready = False
def raised(finger):
    v1 = np.array([finger[0][1] - finger[1][1], finger[0][2] - finger[1][2]])
    v2 = np.array([finger[3][1] - finger[2][1], finger[3][2] - finger[2][2]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    angle_deg = np.degrees(angle_rad)
    if finger[0][0]==0:
        if angle_deg < 90:
            return False
        else:
            return True
    if angle_deg < 150:
        return False
    else:
        return True

while True:
    ret, img = cap.read()
    # img = cv.resize(img, (640, 480), interpolation=cv.INTER_AREA)
    mybody.findbodies(img, draw=True)
    if not start:
        if ret:
            detector.findhands(img)
            ctime = time.time()
            detector.show_fps(img, ctime, ptime)
            ptime = ctime
            coords = detector.getcoord(img)
            fingers = [[], [], [], [], []]
            if coords:
                for i in range(1, 21):
                    fingers[(i - 1) // 4].append(coords[i])
            count = 0
            if coords:
                for i in range(1, 5):
                    if raised(fingers[i]):
                        count += 1

                if fingers[0][2][1] < fingers[0][3][1]:
                    count += 1
            cv.putText(img, str(int(count)) + " fingers are open", (600, 80), cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)
            if count == 2:
                ready = True
                start = True
        ctime = time.time()
        mybody.show_fps(img, ctime, ptime)
        ptime = ctime
        cv.putText(img, f"NOT STARTED", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
        cv.imshow('Video', img)
        cv.waitKey(1)
        continue
    if start and ready:

        ctime = time.time()
        countdown -= ctime-ptime
        ti = math.ceil(countdown)
        mybody.show_fps(img, ctime, ptime)
        ptime = ctime
        cv.putText(img, f"{ti}", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
        cv.imshow('Video', img)
        cv.waitKey(1)
        if ti == 0:
            ready = False
        continue
    ctime = time.time()
    mybody.findbodies(img, draw=True)
    if not ret:
        print("Video ended or error reading frame.")
        break
    # img = cv.resize(frame, (640, 480), interpolation=cv.INTER_AREA)
    mybody.show_fps(img, ctime, ptime)
    lmlist = mybody.getcoords(img)
    if len(lmlist) < 17:
        continue
    right_shoulder = 12
    right_elbow = 14
    right_wrist = 16
    right_arm_angle = mybody.getangle(right_shoulder, right_elbow, right_wrist)
    left_shoulder = 11
    left_elbow = 13
    left_wrist = 15

    left_arm_angle = mybody.getangle(left_shoulder, left_elbow, left_wrist)
    right_hip = 24
    right_ankle = 28
    body_alignment = mybody.getangle(right_shoulder, right_hip, right_ankle)
    for point in [right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, right_hip,
                  right_ankle]:
        cv.circle(img, lmlist[point][1:], 7, (0, 255, 255), -1)
    left_shoulder = lmlist[11][1:]
    right_shoulder = lmlist[12][1:]
    right_elbow = lmlist[14][1:]
    right_wrist = lmlist[16][1:]
    left_elbow = lmlist[13][1:]
    left_wrist = lmlist[15][1:]
    right_hip = lmlist[24][1:]
    right_ankle = lmlist[28][1:]
    prevstate = up
    arm_angle = min(right_arm_angle, left_arm_angle)
    if prevstate != up and prevstate == False:
        minangle = arm_angle
    if prevstate != up and prevstate == True:
        maxangle = arm_angle
        pushupcount += 1
    if arm_angle < 60 :
        up = True
    if arm_angle > 150 :
        up = False
    if prevstate != up and prevstate == False:
        minangle = arm_angle
    if prevstate != up and prevstate == True:
        maxangle = arm_angle
        pushupcount += 1
    if up :
        cv.putText(img, "Go High", (1000, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    else:
        cv.putText(img, "Go Lower!", (1000, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    prevangle = arm_angle
    if body_alignment < 160:
        cv.putText(img, "Keep Your Body Straight!", (0, 500), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
    cv.putText(img, f"R Elbow: {right_arm_angle} deg", (right_elbow[0] - 50, right_elbow[1] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv.putText(img, f"L Elbow: {left_arm_angle} deg", (left_elbow[0] - 50, left_elbow[1] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv.putText(img, f"Body Alignment: {body_alignment} deg", (right_hip[0] - 50, right_hip[1] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    ptime = ctime
    cv.putText(img, f"Push-Ups: {pushupcount}", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)
    cv.imshow('Video', img)
    cv.waitKey(1)