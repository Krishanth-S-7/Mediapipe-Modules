import cv2
import cv2 as cv
import time
import Hand_tracking_module as htm
import numpy as np

cap = cv.VideoCapture(0)
print(cap.isOpened())
detector = htm.HandDetector(number_hands=1,det_con=0.7)
ctime = 0
ptime = 0
# def raised(finger):
#     if finger[3][2] < finger[2][2] < finger[1][2] < finger[0][2]:
#         return True
#     else:
#         return False
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
    if ret:
        detector.findhands(img)
        ctime = time.time()
        detector.show_fps(img, ctime, ptime)
        ptime = ctime
        coords = detector.getcoord(img)
        fingers = [[],[],[],[],[]]
        if coords:
            for i in range(1,21):
                fingers[(i-1)//4].append(coords[i])
        count = 0
        if coords:
            for i in range(1,5):
                if raised(fingers[i]):

                    count += 1

            if fingers[0][2][1]<fingers[0][3][1]:
                count += 1
        cv.putText(img, str(int(count))+" fingers are open", (600, 80), cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5)
    cv2.imshow('Video', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
